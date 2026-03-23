"""
Torch implementation of the SMPL pose max-mixture prior equivalent to
`lib/max_mixture_prior.py` (chumpy version).

Cost implemented:
  min_k [ 0.5 * || L_k (x - mu_k) ||^2 + (-log w'_k) ]
where L_k is the Cholesky factor of the precision (inverse covariance), and
w'_k is the weight scaled by the Gaussian constant and determinants as in the
original implementation.
"""

import numpy as np
import pickle
import torch


class TorchMaxMixturePosePrior:
    def __init__(self, pkl_path, n_gaussians=8, prefix=3, device=None):
        self.pkl_path = pkl_path
        self.n_gaussians = int(n_gaussians)
        self.prefix = int(prefix)
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        with open(self.pkl_path, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        # Float64 pipeline with small jitter for SPD
        covs = gmm['covars'].astype(np.float64)  # (K,69,69)
        means = gmm['means'].astype(np.float64)  # (K,69)
        weights = gmm['weights'].astype(np.float64)  # (K,)

        eye = np.eye(covs.shape[-1], dtype=np.float64)
        covs = np.stack([c + 1e-8 * eye for c in covs], axis=0)

        # Cholesky of covariance to solve Sigma z = diff later
        Lcov = np.stack([np.linalg.cholesky(c) for c in covs], axis=0)  # (K,69,69)

        # Scale the weights as in the original implementation
        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in covs], dtype=np.float64)
        const = (2.0 * np.pi) ** (covs.shape[-1] / 2.0)  # 69-dim
        weights_scaled = weights / (const * (sqrdets / sqrdets.min()))
        weights_scaled = np.clip(weights_scaled, 1e-12, None)
        neg_log_weights = -np.log(weights_scaled)

        # Store as torch tensors on device (float64)
        self.means = torch.from_numpy(means).to(self.device)
        self.Lcov = torch.from_numpy(Lcov).to(self.device)
        self.neg_log_weights = torch.from_numpy(neg_log_weights).to(self.device)

    def __call__(self, pose72_flat):
        """
        pose72_flat: torch.Tensor with shape (72,) on any device.
        Returns: scalar torch.Tensor on the same device as internal params.
        """
        if pose72_flat.ndim != 1 or pose72_flat.numel() != 72:
            raise ValueError('Expected pose72_flat to be a 72-dim 1D tensor')

        x = pose72_flat.to(self.means.device, dtype=self.means.dtype)[self.prefix:]  # (69,)
        if not torch.isfinite(x).all():
            return torch.tensor(float('inf'), device=self.means.device, dtype=pose72_flat.dtype)

        diff = x.unsqueeze(0) - self.means  # (K,69)
        b = diff.unsqueeze(-1)  # (K,69,1)
        # Solve Sigma z = diff using Cholesky factor of Sigma (batch-aware)
        try:
            z = torch.cholesky_solve(b, self.Lcov, upper=False)  # (K,69,1)
        except RuntimeError:
            # Add runtime guard: fallback to generic solve if cholesky_solve fails
            z = torch.linalg.solve(self.Lcov @ self.Lcov.transpose(-1, -2), b)
        sqform = (b.squeeze(-1) * z.squeeze(-1)).sum(dim=1)  # (K,)
        cost_per_comp = 0.5 * sqform + self.neg_log_weights  # (K,)
        # Return on original dtype for downstream math
        return cost_per_comp.min().to(pose72_flat.dtype)


