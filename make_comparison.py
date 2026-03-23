import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
frames = [0, 5, 10, 18]

# Check what extracted frames look like
frame_dir = "results/example_1/hmr2/extracted_frames"
all_frames = sorted(os.listdir(frame_dir)) if os.path.exists(frame_dir) else []
print("Available HMR2 frames:", all_frames[:5])

smplify_dir = "results/example_1/smplify"
smplify_folders = sorted(os.listdir(smplify_dir)) if os.path.exists(smplify_dir) else []
print("Available SMPLify folders:", smplify_folders)

for col, f in enumerate(frames):
    # Row 0: SMPLify
    folder = f"{f:06d}"
    path = f"results/example_1/smplify/{folder}/{folder}_viz.png"
    if os.path.exists(path):
        axes[0, col].imshow(mpimg.imread(path))
        print(f"SMPLify frame {f}: found")
    else:
        axes[0, col].text(0.5, 0.5, f'Missing\n{path}', ha='center', va='center', fontsize=8)
        print(f"SMPLify frame {f}: MISSING at {path}")
    axes[0, col].set_title(f'SMPLify Frame {f}', fontsize=12)
    axes[0, col].axis('off')

    # Row 1: 4DHumans
    if f < len(all_frames):
        path2 = os.path.join(frame_dir, all_frames[f])
        axes[1, col].imshow(mpimg.imread(path2))
        print(f"4DHumans frame {f}: found at {path2}")
    else:
        axes[1, col].text(0.5, 0.5, 'Missing', ha='center', va='center')
        print(f"4DHumans frame {f}: MISSING")
    axes[1, col].set_title(f'4DHumans Frame {f}', fontsize=12)
    axes[1, col].axis('off')

plt.suptitle('SMPLify vs 4DHumans', fontsize=16, fontweight='bold')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/task4_1_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/task4_1_comparison.png")
