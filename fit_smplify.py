"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

============
Code is adapted for EECS 542 / CSE 598 HW5 in University of Michigan.
"""

from os.path import join, exists, abspath, dirname
from os import makedirs
import sys
import logging
import pickle
from time import time
from glob import glob
import argparse
from matplotlib import pyplot as plt
from pathlib import Path

import cv2
import numpy as np
import torch

from smplify_lib.max_mixture_prior import MaxMixtureCompletePrior
from smplify_lib.torch_max_mixture_prior import TorchMaxMixturePosePrior

from hmr2.models.smpl_wrapper import SMPL as SMPL_TORCH
from hmr2.utils.geometry import aa_to_rotmat, perspective_projection
from hmr2.utils.renderer import Renderer

# -------------------- Modern (PyTorch) utils (optional) --------------------
def _build_renderer_cfg(focal_length=5000.0, image_size=256):
    class _Obj:
        pass
    cfg = _Obj()
    cfg.EXTRA = _Obj()
    cfg.MODEL = _Obj()
    cfg.EXTRA.FOCAL_LENGTH = float(focal_length)
    cfg.MODEL.IMAGE_SIZE = int(image_size)
    cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    return cfg



# Single environment check and device definition
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------- Dataset mappings --------------------
def get_dataset_torso_mappings(dataset):
    """Return dataset-specific torso and shoulder/hip mappings for camera init.

    Returns a dict with keys:
      - torso_src_ids: indices of torso joints in the source 2D keypoints
      - torso_smpl_ids: indices of torso joints in SMPL joints
      - src_shoulder_hip_pairs: two pairs [L_shoulder, L_hip], [R_shoulder, R_hip] in source 2D
      - smpl_shoulder_hip_pairs: two pairs [L_shoulder, L_hip], [R_shoulder, R_hip] in SMPL joints
    """
    ds = (dataset or 'lsp').lower()
    if ds == 'lsp':
        return {
            'torso_src_ids': [2, 3, 8, 9],
            'torso_smpl_ids': [2, 1, 17, 16],
            'src_shoulder_hip_pairs': ([9, 3], [8, 2]),
            'smpl_shoulder_hip_pairs': ([16, 1], [17, 2]),
        }
    elif ds == 'openpose':
        return {
            'torso_src_ids': [9, 12, 2, 5],
            'torso_smpl_ids': [2, 1, 17, 16],
            'src_shoulder_hip_pairs': ([5, 12], [2, 9]),
            'smpl_shoulder_hip_pairs': ([16, 1], [17, 2]),
        }
    raise ValueError(f"Unsupported dataset for init mappings: {dataset}")


def get_dataset_opt_mappings(dataset):
    """Return dataset-specific mappings for optimize_on_joints.

    Returns a dict with keys:
      - src_ids: list of source 2D indices to supervise
      - smpl_ids: list of SMPL joint indices corresponding to the first 12 source joints
      - head_vertex_id: vertex id to represent head top in SMPL
      - base_weights: basic per-joint weights for reprojection loss (len == len(src_ids))
    """
    ds = (dataset or 'lsp').lower()
    if ds == 'lsp':
        src_ids = list(range(12)) + [13]    # 13 is the head top
        smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
        head_vertex_id = 411
        # the definition of hips in SMPL and LSP is significantly different so set
        # their weights to zero.
        base_weights = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        return {
            'src_ids': src_ids,
            'smpl_ids': smpl_ids,
            'head_vertex_id': head_vertex_id,
            'base_weights': base_weights,
        }
    elif ds == 'openpose':
        src_ids = list(range(1, 15)) + [0] # 0 is the head top
        smpl_ids = [12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7]
        head_vertex_id = 411
        base_weights = np.ones(len(src_ids), dtype=np.float32)
        return {
            'src_ids': src_ids,
            'smpl_ids': smpl_ids,
            'head_vertex_id': head_vertex_id,
            'base_weights': base_weights,
        }
    else:
        raise ValueError(f"Unsupported dataset for optimize mappings: {dataset}")

    # Mapping from LSP joints to SMPL joints.
    # 0 Right ankle  8
    # 1 Right knee   5
    # 2 Right hip    2
    # 3 Left hip     1
    # 4 Left knee    4
    # 5 Left ankle   7
    # 6 Right wrist  21
    # 7 Right elbow  19
    # 8 Right shoulder 17
    # 9 Left shoulder  16
    # 10 Left elbow    18
    # 11 Left wrist    20
    # 12 Neck           -
    # 13 Head top       added


def scale_img_and_j2d(img, j2d, scale_factor):
    """Scale the image and the joints by the scale factor."""
    img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor))
    j2d = j2d * scale_factor
    return img, j2d


def guess_init(model, focal_length, j2d, init_pose, init_map):
    """Estimate an initial camera translation using torso size via similar triangles.
    Provides a robust first estimate of the camera translation along z (tz)
    by comparing average torso (shoulder-hip) length in 3D (from SMPL posed
    with init_pose) to the corresponding distance in 2D detections.

    Parameters:
        model (SMPL_TORCH): SMPL model on DEVICE used to compute posed 3D joints.
        focal_length (float): Focal length in pixels used for depth estimation.
        j2d (np.ndarray): Array of 2D joints, shape (J, 2), in dataset ordering.
        init_pose (np.ndarray): 72-D axis-angle pose used to pose SMPL and measure
            3D torso length. First 3 dims are global orient, remaining 69 are body.
        init_map (dict): Dataset torso mapping as returned by
            get_dataset_torso_mappings(dataset). Must define
            'src_shoulder_hip_pairs' and 'smpl_shoulder_hip_pairs'.

    Returns:
        np.ndarray: Translation vector [tx, ty, tz] with tx=ty=0 and estimated tz.
    """
    smpl_t = model

    init_pose = torch.from_numpy(init_pose.astype(np.float32)).to(DEVICE)
    global_orient_aa = init_pose[:3].view(1, 3)
    body_pose_aa = init_pose[3:].view(1, 23, 3)
    glob_rotmat = aa_to_rotmat(global_orient_aa.view(-1, 3)).view(1, 1, 3, 3)
    body_rotmat = aa_to_rotmat(body_pose_aa.view(-1, 3)).view(1, 23, 3, 3)
    out = smpl_t(global_orient=glob_rotmat, body_pose=body_rotmat, betas=torch.zeros(1,10,device=DEVICE), pose2rot=False)
    J = out.joints[0]

    # Use dataset-specific shoulder/hip pairs
    (smpl_L_sh, smpl_L_hip), (smpl_R_sh, smpl_R_hip) = init_map['smpl_shoulder_hip_pairs']
    diff3d = torch.stack([J[smpl_L_sh] - J[smpl_L_hip], J[smpl_R_sh] - J[smpl_R_hip]], dim=0)
    mean_height3d = torch.linalg.norm(diff3d, dim=1).mean().item()

    # Source 2D indices for shoulder/hip pairs
    (src_L_sh, src_L_hip), (src_R_sh, src_R_hip) = init_map['src_shoulder_hip_pairs']
    j2d = j2d.astype(np.float32)
    diff2d = np.stack([j2d[src_L_sh] - j2d[src_L_hip], j2d[src_R_sh] - j2d[src_R_hip]], axis=0)
    mean_height2d = np.linalg.norm(diff2d, axis=1).mean()

    est_d = focal_length * (mean_height3d / (mean_height2d + 1e-9))
    return np.array([0., 0., est_d], dtype=np.float32)


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=5000.,
                      viz=False,
                      viz_dir=None,
                      viz_prefix=None,
                      dataset='lsp'):
    """Optimize initial camera translation and global body orientation.
    Uses a short optimization to align the torso (shoulders/hips) by jointly
    refining the camera translation (t) and the global orientation (first 3
    pose dims) while keeping the body pose fixed to init_pose. This provides a
    good starting point for the full pose-and-shape optimization.

    Parameters:
        model (SMPL_TORCH): SMPL model on DEVICE.
        j2d (np.ndarray): 2D joints array (J, 2) in the dataset's keypoint order.
        img (np.ndarray): Input image (H, W, 3) in BGR (as read by OpenCV). Used
            for visualization and to compute image center.
        init_pose (np.ndarray): 72-D axis-angle pose to start with before optimization.
            The first 3 values are global orient, the remaining 69 are body pose.
        flength (float, optional): Focal length in pixels. Defaults to 5000.
        viz (bool, optional): If True, saves intermediate visualizations.
        viz_dir (str | Path, optional): Directory to save visualizations.
        viz_prefix (str, optional): Filename prefix for saved figures.
        dataset (str, optional): Dataset name to pick torso mappings.

    Returns:
        tuple:
            - cam (dict): {'f': np.array([fx, fy]), 'c': np.array([cx, cy]), 't': np.array([tx, ty, tz])}
            - body_orient_aa (np.ndarray): (3,) optimized global orientation (axis-angle)
    """
    smpl_t = model

    H, W = img.shape[0], img.shape[1]
    center = np.array([W / 2., H / 2.], dtype=np.float32)

    # Dataset-specific torso indices
    torso_map = get_dataset_torso_mappings(dataset)
    torso_src_ids = torso_map['torso_src_ids']
    torso_smpl_ids = torso_map['torso_smpl_ids']

    # Initialize camera translation using guess_init
    init_t = guess_init(model, flength, j2d, init_pose, torso_map)
    lr = 0.05

    #########################################################
    # TODO Task 3.2
    # General steps: 
    # 1. Create a torch tensor called `cam_t` for the camera translation (from init_t),
    # and a torch tensor called `body_orient_aa` for the global body orientation in axis-angle.
    # The global body orientation should be initialized as zeros.
    # Remember to set requires_grad=True for both tensors.
    #
    # 2. Next create a torch optimizer with the parameters [cam_t, body_orient_aa] and learning rate lr.
    # You can use the torch.optim.Adam.
    #########################################################
    # Your code here.
    cam_t = torch.tensor(init_t.reshape(1, 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
    body_orient_aa = torch.zeros(1, 3, dtype=torch.float32, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([cam_t, body_orient_aa], lr=lr)

    #########################################################


    j2d_torso_gt = torch.from_numpy(j2d[torso_src_ids].astype(np.float32)).to(DEVICE).unsqueeze(0)
    for i in range(1001):
        #########################################################
        # TODO Task 3.2 (continued)
        # General steps:
        # 1. Zero the gradients of the optimizer.
        # 2. Call smpl_t to get the 3D joints. Check out how we call it in guess_init() or in render_smpl_tpose.py.
        # You may find aa_to_rotmat() useful to convert axis-angle to rotation matrix.
        # For the output joints, you can select torso joints by using the torso_smpl_ids.
        # 
        # 3. Calculate the projected 2D joints called `proj2d` by projecting the selected 3D joints using perspective_projection(). 
        # Please check out its source code in hmr2/utils/geometry.py. Camera center and focal length are provided to you.
        #
        # 4. Compute the reprojection loss called `loss_reproj` between the projected 2D joints and the ground truth 2D joints.
        # You can do so by calculating ((proj2D - j2d_torso_gt) ** 2).mean().
        #########################################################
        # Your code here.
        optimizer.zero_grad()
        glob_rotmat = aa_to_rotmat(body_orient_aa.view(-1, 3)).view(1, 1, 3, 3)
        body_rotmat_fixed = aa_to_rotmat(
            torch.from_numpy(init_pose[3:].astype(np.float32)).to(DEVICE).view(-1, 3)
        ).view(1, 23, 3, 3)
        out = smpl_t(global_orient=glob_rotmat, body_pose=body_rotmat_fixed,
                     betas=torch.zeros(1, 10, device=DEVICE), pose2rot=False)
        joints_3d = out.joints[:, torso_smpl_ids, :]
        proj2d = perspective_projection(
            points=joints_3d,
            translation=cam_t,
            focal_length=torch.tensor([[flength, flength]], dtype=torch.float32, device=DEVICE),
            camera_center=torch.tensor([[center[0], center[1]]], dtype=torch.float32, device=DEVICE),
        )
        loss_reproj = ((proj2d - j2d_torso_gt) ** 2).mean()

        #########################################################

        # Add a regularization term to penalize large changes in the camera translation
        loss_depth = 100 * (cam_t[:, 2] - float(init_t[2])) ** 2
        loss = loss_reproj + loss_depth
        if i % 10 == 0:
            print(f"init_cam step={i} "
                  f"reproj={float(loss_reproj.detach().cpu().item()):.6f} "
                  f"depth={float(loss_depth.mean().detach().cpu().item()):.6f} "
                  f"total={float(loss.detach().cpu().item()):.6f}")
        loss.backward()
        optimizer.step()

        if viz and (i % 100 == 0):
            # visualize GT vs optimized torso joints using matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img[:, :, ::-1])
            proj2d_np = proj2d.detach().cpu().numpy().squeeze(0)
            gt_np = j2d[torso_src_ids]
            ax.scatter(gt_np[:, 0], gt_np[:, 1], s=20, c='lime', edgecolors='k', linewidths=0.5, label='GT torso')
            ax.scatter(proj2d_np[:, 0], proj2d_np[:, 1], s=16, c='red', edgecolors='k', linewidths=0.4, label='Optimized torso')
            ax.set_axis_off()
            ax.legend(loc='lower right')
            fig.tight_layout()
            out_path = join(viz_dir, f"{viz_prefix}_initcam_step{i:03d}.png")
            fig.savefig(out_path)
            plt.close(fig)

    cam_np_t = cam_t.detach().cpu().numpy().reshape(3)
    body_orient_np = body_orient_aa.detach().cpu().numpy().reshape(3)

    # Simple camera dictionary
    cam = {
        'f': np.array([flength, flength], dtype=np.float32),
        'c': center,
        't': cam_np_t,
    }
    return (cam, body_orient_np)



def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       body_orient,
                       n_betas=10,
                       conf=None,
                       viz=False,
                       viz_dir=None,
                       viz_prefix=None,
                       dataset='lsp',
                       reproj_only=False):
    """Optimize SMPL pose and shape to match 2D keypoints for a single image.

    Parameters:
        j2d (np.ndarray): 2D keypoints in the dataset ordering, shape (J, 2 or 3).
            Only the first two columns (x, y) are used; confidences passed via conf.
        model (SMPL_TORCH): SMPL model (on DEVICE) used to generate 3D joints/vertices.
        cam (dict): Camera parameters with keys:
            - 'f': np.ndarray [fx, fy]
            - 'c': np.ndarray [cx, cy]
            - 't': np.ndarray [tx, ty, tz]
        img (np.ndarray): Input image (H, W, 3), used for optional visualization.
        prior: Prior object used to extract a mean body pose (e.g., GMM prior).
        body_orient (np.ndarray): Initial global orientation axis-angle, shape (3,).
        n_betas (int): Number of SMPL shape coefficients to optimize.
        conf (np.ndarray | None): Per-joint confidence scores aligned to j2d; if
            provided, used to weight the reprojection loss.
        viz (bool): If True, saves intermediate optimization visualizations.
        viz_dir (str | Path | None): Directory to save visualizations.
        viz_prefix (str | None): Prefix for saved visualization filenames.
        dataset (str): Dataset name used to select joint mappings (e.g., 'lsp').
        reproj_only (bool): If True, optimize with only reprojection loss; if False,
            include angle, pose, and shape priors.

    Returns:
        tuple:
            - fit_out (dict): Fitted outputs {'verts': (V,3), 'pose': (72,), 'betas': (n_betas,)}
            - opt_j2d (np.ndarray): Projected 2D joints used for supervision, shape (K, 2)
            - losses (dict): Logged losses per stage
    """
    # Use the passed-in SMPL model
    smpl_t = model
    # Torch prior (max mixture of Gaussians over body pose)
    prior_torch = TorchMaxMixturePosePrior(PRIOR_PKL_PATH, n_gaussians=8, prefix=3, device=DEVICE)
    
    # Set up camera parameters.
    focal = torch.tensor([[float(cam['f'][0]), float(cam['f'][1])]], dtype=torch.float32, device=DEVICE)
    cam_center = torch.tensor([[float(cam['c'][0]), float(cam['c'][1])]], dtype=torch.float32, device=DEVICE)
    cam_t_fixed = torch.tensor(cam['t'].reshape(1, 3), dtype=torch.float32, device=DEVICE)

    # Dataset-specific joint mapping.
    # Since SMPL does not have a joint on the top of the head, we need to select a vertex on the head as the delegate joint.
    opt_map = get_dataset_opt_mappings(dataset)
    # Indices for the source/pseudo-gt/target joints. Use this to select the joints/confidence scores from the input j2d.
    src_ids = opt_map['src_ids']
    # Indices for the SMPL joints.
    smpl_ids = opt_map['smpl_ids']
    head_vertex_id = opt_map['head_vertex_id']
    # Base weights to disable joints that are defined too differently in SMPL and the supervision.
    base_weights = opt_map['base_weights']
    

    # Initialize body pose from GMM prior mean (no global orient)
    try:
        mean_pose = prior.weights.dot(prior.means)
        mean_pose = mean_pose.r if hasattr(mean_pose, 'r') else np.asarray(mean_pose)
    except Exception:
        # Fallback to zeros if prior is unavailable
        mean_pose = np.zeros(69, dtype=np.float32)

    #########################################################
    # TODO Task 3.4
    # General steps:
    # 1. Compute joint-wise weights for the reprojection loss so we don't try 
    # to match joints with low confidence too much.
    # You can do so by multiplying the base weights by the confidence scores.
    #
    # 2. Create variables for optimization:
    # - `betas` for shape coefficients
    # - `body_pose_aa` for body pose in axis-angle (initialized from the mean pose)
    # - `global_orient_aa` for global orientation in axis-angle
    # Remember to set requires_grad=True for all variables.
    #########################################################
    # Your code here.
    conf_scores = conf[src_ids].astype(np.float32) if conf is not None else np.ones(len(src_ids), dtype=np.float32)
    weights = torch.tensor(base_weights * conf_scores, dtype=torch.float32, device=DEVICE)

    betas = torch.zeros(1, n_betas, dtype=torch.float32, device=DEVICE, requires_grad=True)
    body_pose_aa = torch.tensor(mean_pose.reshape(1, 23, 3), dtype=torch.float32, device=DEVICE, requires_grad=True)
    global_orient_aa = torch.tensor(body_orient.reshape(1, 3), dtype=torch.float32, device=DEVICE, requires_grad=True)

    #########################################################

    # Targets
    j2d_target = torch.from_numpy(j2d[src_ids].astype(np.float32)).to(DEVICE)  # (13,2)

    # Optimization schedule
    if reproj_only:
        stages = {
            'w_reproj': [0.01], 'w_pose': [0.0], 'w_angle': [0.0],
            'w_betas': [0.0], 'lr': [0.005], 'steps': [1000]
        }
        num_stages = len(stages['w_reproj'])
        losses = {
            'reproj': [[] for _ in range(num_stages)],
            'angle': [[] for _ in range(num_stages)],
            'shape': [[] for _ in range(num_stages)],
            'pose': [[] for _ in range(num_stages)],
            'total': [[] for _ in range(num_stages)],
            'step': [[] for _ in range(num_stages)]
        }
    else:
        stages = {
            #########################################################
            # TODO Task 3.5
            # Adjust the weights for different loss terms.
            # We recommend you to debug on just one image!
            # Some hints (assuming w_reproj = 0.01 for all stages):
            # - You are recommended to leave w_reproj unchanged to measure the the absolute performance of joint matching.
            # - w_pose can roughly be in the range of 1-10 for the first two stages, 
            #   <1 for the third stage, and even smaller for the fourth stage.
            # - w_angle can be the exact same as w_pose since they can be both counted as some sort of pose prior.
            # - w_betas can be very large in the beginning (recommended range: (1, 100)) and then gradually decreases to 1.
            # - You can adjust learning rates or number of steps for different stages as you see fit.
            #########################################################
            'w_reproj': [0.01,  0.01,  0.01,  0.01],
            'w_pose':   [8.0,   4.0,   0.5,   0.05],
            'w_angle':  [8.0,   4.0,   0.5,   0.05],
            'w_betas':  [50.0,  20.0,  5.0,   1.0 ],
            'lr':       [0.01,  0.01,  0.005, 0.005],
            'steps':    [500,   1000,  5000,  5000 ]
        }
        num_stages = len(stages['w_reproj'])
        losses = {
            'reproj': [[] for _ in range(num_stages)],
            'angle': [[] for _ in range(num_stages)],
            'shape': [[] for _ in range(num_stages)],
            'pose': [[] for _ in range(num_stages)],
            'total': [[] for _ in range(num_stages)],
            'step': [[] for _ in range(num_stages)]
        }

    # Single param list for optimizer
    optimizer = torch.optim.Adam([
        {"params": [betas, body_pose_aa, global_orient_aa], "lr": stages['lr'][0]},
    ])

    for stage_idx in range(num_stages):
        w_reproj = stages['w_reproj'][stage_idx]
        w_pose = stages['w_pose'][stage_idx]
        w_angle = stages['w_angle'][stage_idx]
        w_betas = stages['w_betas'][stage_idx]
        steps = stages['steps'][stage_idx]
        # update learning rate for the current stage
        optimizer.param_groups[0]['lr'] = stages['lr'][stage_idx]
        for i in range(steps):
            #########################################################
            # TODO Task 3.4 (continued)
            # General steps:
            # 1. Convert the global and body pose to rotation matrices (using aa_to_rotmat()).
            # 
            # 2. Call smpl_t to get the 3D joints.
            # 
            # 3. Select 3D joints by smpl_ids, and select the head vertex by head_vertex_id.
            # You can get vertices by calling out.vertices if `out` is the SMPL output.
            # Then concatenate the 3D joints and the head vertex to get the whole 3D joints we are gonna match.
            # 
            # 4. Project the 3D joints to 2D using perspective_projection().
            # 
            # 5. Compute the reprojection loss (Note they are weighted by the joint weights).
            # You can do so by calculating ((projected_joints - j2d_target) ** 2).mean().
            #########################################################
            # Your code here.

            # Losses
            optimizer.zero_grad()
            glob_rotmat = aa_to_rotmat(global_orient_aa.view(-1, 3)).view(1, 1, 3, 3)
            body_rotmat = aa_to_rotmat(body_pose_aa.view(-1, 3)).view(1, 23, 3, 3)
            out = smpl_t(global_orient=glob_rotmat, body_pose=body_rotmat, betas=betas, pose2rot=False)
            joints_sel = out.joints[:, smpl_ids, :]
            head_v = out.vertices[:, head_vertex_id:head_vertex_id+1, :]
            joints_3d = torch.cat([joints_sel, head_v], dim=1)
            proj2d = perspective_projection(
                points=joints_3d,
                translation=cam_t_fixed,
                focal_length=focal,
                camera_center=cam_center,
            ).squeeze(0)
            diff = proj2d - j2d_target
            loss_reproj = w_reproj * (weights.unsqueeze(1) * diff ** 2).mean()
            #########################################################

            if reproj_only:
                loss_angle_prior = torch.tensor(0.0, device=DEVICE)
                loss_shape_prior = torch.tensor(0.0, device=DEVICE)
                loss_pose_prior = torch.tensor(0.0, device=DEVICE)
            else:
                # Angle prior, penalizing unnatural bending of albows and knees.
                pose_vec = torch.cat([global_orient_aa.view(-1), body_pose_aa.view(-1)])  # (72,)
                loss_angle_prior = w_angle * (torch.exp(pose_vec[55]) 
                                              + torch.exp(-pose_vec[58])
                                              + torch.exp(-pose_vec[12])
                                              + torch.exp(-pose_vec[15]))

                # Shape prior
                loss_shape_prior = w_betas * (betas ** 2).sum()
                
                # Max-mixture GMM pose prior
                mix_prior_val = prior_torch(pose_vec)
                loss_pose_prior = w_pose * mix_prior_val

            loss = loss_reproj + loss_angle_prior + loss_shape_prior + loss_pose_prior
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                loss_reproj_ = loss_reproj.detach().cpu().item()
                loss_angle_prior_ = loss_angle_prior.detach().cpu().item()
                loss_shape_prior_ = loss_shape_prior.detach().cpu().item()
                loss_pose_prior_ = loss_pose_prior.detach().cpu().item()
                loss_ = loss.detach().cpu().item()
                losses['reproj'][stage_idx].append(loss_reproj_)
                losses['angle'][stage_idx].append(loss_angle_prior_)
                losses['shape'][stage_idx].append(loss_shape_prior_)
                losses['pose'][stage_idx].append(loss_pose_prior_)
                losses['total'][stage_idx].append(loss_)
                losses['step'][stage_idx].append(i)
                print(f"opt_joints stage={stage_idx} step={i} "
                      f"reproj={loss_reproj_:.6f} angle={loss_angle_prior_:.6f} "
                      f"shape={loss_shape_prior_:.6f} pose={loss_pose_prior_:.6f} "
                      f"total={loss_:.6f}")

            save_img_interval = steps // 10
            if viz and (i % save_img_interval == 0):
                # visualize GT vs optimized joints using matplotlib
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img[:, :, ::-1])
                proj2d_np = proj2d.detach().cpu().numpy()
                gt_np = j2d_target.detach().cpu().numpy()
                ax.scatter(gt_np[:, 0], gt_np[:, 1], s=20, c='lime', edgecolors='k', linewidths=0.5, label='GT joints')
                ax.scatter(proj2d_np[:, 0], proj2d_np[:, 1], s=16, c='red', edgecolors='k', linewidths=0.4, label='Optimized joints')
                ax.set_axis_off()
                ax.legend(loc='lower right')
                fig.tight_layout()
                out_path = join(viz_dir, f"{viz_prefix}_opt_stage{stage_idx}_iter{i:03d}.png")
                fig.savefig(out_path)
                plt.close(fig)

    # Final outputs
    with torch.no_grad():
        glob_rotmat = aa_to_rotmat(global_orient_aa.view(-1, 3)).view(1, 1, 3, 3)
        body_rotmat = aa_to_rotmat(body_pose_aa.view(-1, 3)).view(1, 23, 3, 3)
        out = smpl_t(global_orient=glob_rotmat, body_pose=body_rotmat, betas=betas, pose2rot=False)
        verts = out.vertices.squeeze(0).cpu().numpy()
        pose_aa = torch.cat([global_orient_aa.view(1, 3), body_pose_aa.view(23, 3)], dim=0).reshape(-1).cpu().numpy()
        betas_np = betas.squeeze(0).cpu().numpy()

    # Provide projected joints as opt_j2d (12 SMPL joints + head vertex)
    sel3d_final = torch.cat([
        out.joints[:, smpl_ids, :],
        out.vertices[:, head_vertex_id:head_vertex_id+1, :]
    ], dim=1)
    opt_j2d = perspective_projection(points=sel3d_final,
                                     translation=cam_t_fixed,
                                     focal_length=focal,
                                     camera_center=cam_center).squeeze(0).detach().cpu().numpy()
    fit_out = { 'verts': verts, 'pose': pose_aa, 'betas': betas_np }
    return (fit_out, opt_j2d, losses)


def run_single_fit(img,
                   j2d,
                   conf,
                   model,
                   n_betas=10,
                   flength=5000.,
                   scale_factor=1,
                   viz=False,
                   viz_dir=None,
                   viz_prefix=None,
                   render_degrees=None,
                   dataset='lsp',
                   reproj_only=False):
    """Run the fit for one specific image.
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param render_degrees: list of degrees in azimuth to render the final fit when saving results
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """
    if render_degrees is None:
        render_degrees = []

    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(PRIOR_PKL_PATH, n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))    # (72)

    if scale_factor != 1:
        img, j2d = scale_img_and_j2d(img, j2d, scale_factor)

    # estimate the camera parameters
    (cam, body_orient) = initialize_camera(
        model,
        j2d,
        img,
        init_pose,
        flength=flength,
        viz=viz,
        viz_dir=viz_dir,
        viz_prefix=viz_prefix,
        dataset=dataset)
    # sys.exit()  # TODO: remove this after you finish Task 3.2

    # fit
    (fit_out, opt_j2d, losses) = optimize_on_joints(
        j2d,
        model,
        cam,
        img,
        prior,
        body_orient,
        n_betas=n_betas,
        conf=conf,
        viz=viz,
        viz_dir=viz_dir,
        viz_prefix=viz_prefix,
        dataset=dataset,
        reproj_only=reproj_only)

    h = img.shape[0]
    w = img.shape[1]

    images = []
    orig_v = fit_out['verts']
    faces = model.faces
    cfg_r = _build_renderer_cfg(focal_length=flength, image_size=max(h, w))
    renderer = Renderer(cfg_r, faces=faces)
    cam_t_np = cam['t'].copy()
    for deg in render_degrees:
        rgba = renderer.render_rgba(
            vertices=orig_v,
            mesh_base_color=(1.0, 0.4, 0.7),  # pink (DO NOT CHANGE)
            cam_t=cam_t_np,
            rot_axis=[0,1,0],
            rot_angle=deg,
            render_res=[w, h],
            scene_bg_color=(1,1,1),
        )
        # Composite RGBA render over input image
        input_img = img.astype(np.float32)[:, :, ::-1] / 255.0
        alpha = rgba[:, :, 3:]
        overlay = input_img * (1 - alpha) + rgba[:, :, :3] * alpha
        im = (overlay * 255.0).astype('uint8')
        images.append(im)

    # return fit parameters
    params = {'cam_t': cam['t'],
              'f': cam['f'],
              'pose': fit_out['pose'],
              'betas': fit_out['betas'],
              'opt_j2d': opt_j2d,
              'losses': losses}

    return params, images


def main(img_dir,
         joints_dir,
         out_dir,
         n_betas=10,
         flength=5000.,
         viz=True,
         dataset='lsp',
         selected_indices='',
         reproj_only=False):
    """Set up paths to image and joint data, saves results.
    :param img_dir: folder containing input images
    :param joints_dir: folder containing joints .npy files
    :param out_dir: output folder
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (an estimate)
    :param viz: boolean, if True enables visualization during optimization
    :param selected_indices: comma-separated indices to process (string)
    """

    img_dir = Path(img_dir)
    joints_dir = Path(joints_dir)
    if not exists(out_dir):
        makedirs(out_dir)

    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    render_degrees = [0.]
    scale_factor = 2

    # Use gender-neutral model
    model = SMPL_TORCH(model_path=MODEL_NEUTRAL_PATH, gender='NEUTRAL',
                       num_betas=n_betas, use_vanilla_joints=True).to(DEVICE)

    # Load images and matching GT joints
    img_paths = sorted([
        *img_dir.glob('*.png'),
        *img_dir.glob('*.jpg'),
        *img_dir.glob('*.jpeg'),
    ])
    joints_paths = sorted([*joints_dir.glob('*.npy')])
    assert len(img_paths) == len(joints_paths), "Number of images and joints must match"
    joints_list = []
    for joints_path in joints_paths:
        joints = np.load(joints_path).T
        joints_list.append(joints)
    joints_all = np.stack(joints_list, axis=0)

    # Parse selected indices (if provided)
    selected_set = []
    if selected_indices:
        selected_indices = str(selected_indices)
        toks = [t.strip() for t in selected_indices.split(',') if t.strip() != '']
        if len(toks) > 0:
            try:
                selected_set = [int(t) for t in toks]
            except Exception:
                raise ValueError("selected_indices must be a comma-separated list of integers, e.g., '0,2,5'")

    for ind, img_path in enumerate(img_paths):
        if len(selected_set) > 0 and ind not in selected_set:
            continue
        img_out_dir = Path(out_dir) / f"{ind:06d}"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        pkl_out_path = str(img_out_dir / f'{ind:06d}.pkl')

        print(f'Fitting 3D body on `{img_path}` (saving to `{img_out_dir}`).')
        img = cv2.imread(str(img_path))

        # Retrieve joints and confidence for this image
        joints = joints_all[ind][:, :2]
        conf = joints_all[ind][:, 2]

        params, vis = run_single_fit(
            img,
            joints,
            conf,
            model,
            n_betas=n_betas,
            flength=flength,
            scale_factor=scale_factor,
            viz=viz,
            viz_dir=img_out_dir,
            viz_prefix=f"{ind:06d}",
            render_degrees=render_degrees,
            dataset=dataset,
            reproj_only=reproj_only)

        #########################################################
        # TODO Task 4.1
        # You might want to change or adopt the following code to only visualize
        # the input image overlayed with rendered SMPL mesh.
        #########################################################
        if viz:
            img, joints = scale_img_and_j2d(img, joints, scale_factor)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # Left: input with GT 2D joints
            ax[0].imshow(img[:, :, ::-1])
            ax[0].scatter(joints[:, 0], joints[:, 1], s=20, c='lime', edgecolors='k', linewidths=0.5)
            # Overlay optimized 2D reprojections in red
            opt_j2d = params.get('opt_j2d', None)
            if opt_j2d is not None:
                ax[0].scatter(opt_j2d[:, 0], opt_j2d[:, 1], s=16, c='red', edgecolors='k', linewidths=0.4)
            ax[0].set_title('Input + 2D joints')
            ax[0].set_axis_off()
            # Right: input with predicted mesh overlay
            ax[1].imshow(vis[0])
            ax[1].set_title('Input + predicted mesh')
            ax[1].set_axis_off()
            fig.tight_layout()
            viz_path = pkl_out_path.replace('.pkl', '_viz.png')
            fig.savefig(viz_path, dpi=200)
            plt.close(fig)

        with open(pkl_out_path, 'wb') as outf:
            pickle.dump(params, outf)

        # Save loss curves as a 4x5 grid (rows=stages, cols=metrics)
        losses = params['losses']
        metrics = ['reproj', 'angle', 'shape', 'pose', 'total']
        num_stages = len(params['losses']['step'])
        fig, axes = plt.subplots(num_stages, len(metrics), figsize=(len(metrics) * 3.0, num_stages * 2.5))
        for si in range(num_stages):
            for mi, m in enumerate(metrics):
                ax = axes[si, mi] if num_stages > 1 else axes[mi]
                steps = losses['step'][si]
                vals = losses[m][si]
                ax.plot(steps, vals, linewidth=1.0)
                if si == 0:
                    ax.set_title(m)
                if mi == 0:
                    ax.set_ylabel(f'Stage {si}')
                if si == (num_stages - 1):
                    ax.set_xlabel('step')
                ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        plt_path = pkl_out_path.replace('.pkl', '_losses.png')
        fig.savefig(plt_path, dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        '--img_dir',
        default='example_data/images/lsp',
        help="Input directory containing images")
    parser.add_argument(
        '--joints_dir',
        default='example_data/images/lsp',
        help="Input directory containing joints .npy files")
    parser.add_argument(
        '--selected_indices',
        default='',
        help="Comma-separated indices to process (e.g., '0,3,5'). Empty = all")
    parser.add_argument(
        '--dataset',
        default='lsp',
        help='Dataset name for keypoint mappings (e.g., lsp)')
    parser.add_argument(
        '--out_dir',
        default='results/lsp/smplify',
        type=str,
        help='Where results will be saved')
    parser.add_argument(
        '--model_dir',
        default='data/',
        help='Directory containing the SMPL models')
    parser.add_argument(
        '--n_betas',
        default=10,
        type=int,
        help="Specify the number of shape coefficients to use.")
    parser.add_argument(
        '--flength',
        default=5000,
        type=float,
        help="Specify value of focal length.")
    parser.add_argument(
        '--side_view_thsh',
        default=25,
        type=float,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--viz',
        default=False,
        action='store_true',
        help="Turns on visualization of intermediate optimization steps "
        "and final results.")
    parser.add_argument(
        '--reproj_only',
        default=False,
        action='store_true',
        help='Use only reprojection loss (disable angle/pose/shape priors)')
    args = parser.parse_args()

    print('Using gender neutral SMPL model.')

    # Set up paths & load models.
    # Model paths:
    MODEL_NEUTRAL_PATH = join(args.model_dir, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    # MODEL_FEMALE_PATH = join(args.model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    # MODEL_MALE_PATH = join(args.model_dir, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    PRIOR_PKL_PATH = join(args.model_dir, 'gmm_08.pkl')

    main(args.img_dir, args.joints_dir, args.out_dir, args.n_betas,
         args.flength, args.viz, args.dataset, args.selected_indices, args.reproj_only)
