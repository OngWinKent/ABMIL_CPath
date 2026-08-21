import warnings
warnings.filterwarnings("ignore")
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from options import get_parse_args
import lmdb
import pickle
from PIL import Image
import io
from datasets import build_dataloader
from modules import build_model
import matplotlib.pyplot as plt

"""PyTorch Dataset to load WSI features, slide labels, patch labels, and coordinates."""
class SlideDataset(Dataset):
    def __init__(self, root_dir: str, dataset_name: str):
        root_dir = Path(root_dir) / dataset_name
        csv_path = root_dir / f"{dataset_name}.csv"

        self.coords_dir = root_dir / "coords"
        self.patch_labels_dir = root_dir / "patch_labels"
        self.features_dir = root_dir / "uni" / "pt_files"
        self.label_col = "label"

        df_raw = pd.read_csv(csv_path)

        # Pre-filter dataset: Keep only slides where ALL required files exist
        valid_indices = []
        for idx, row in df_raw.iterrows():
            slide_id = Path(str(row["slide"])).stem

            # Check required files
            feat_path = self.features_dir / f"{slide_id}.pt"
            coords_path = self.coords_dir / f"{slide_id}.npy"
            patch_label_npy = self.patch_labels_dir / f"{slide_id}.npy"

            # Must have features, coords, and patch_label
            if (
                feat_path.exists()
                and coords_path.exists()
                and patch_label_npy.exists()
            ):
                valid_indices.append(idx)

        # Filter dataframe and reset index
        self.df = df_raw.iloc[valid_indices].reset_index(drop=True)

        print(
            f"[Dataset] Loaded {len(self.df)} / {len(df_raw)} valid slides (Skipped {len(df_raw) - len(self.df)} due to missing files)."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        raw_slide = str(self.df.iloc[idx]["slide"])
        slide_id = Path(raw_slide).stem

        # 1. Load Features (.pt file)
        feat_path = self.features_dir / f"{slide_id}.pt"
        features = torch.load(feat_path, weights_only=True)

        # 2. Slide-level Label (from CSV)
        label = torch.tensor(self.df.iloc[idx][self.label_col])

        # 3. Patch-level Labels (.npy file with robust 0D/dict unwrapping)
        patch_label_path = self.patch_labels_dir / f"{slide_id}.npy"
        patch_label = np.load(patch_label_path)

        # 4. Patch Coordinates (.npy file)
        coords_path = self.coords_dir / f"{slide_id}.npy"
        coords = np.load(coords_path)

        return slide_id, features, label, patch_label, coords

"""Load every patch belonging to one slide from LMDB."""
def load_slide_patches(txn: lmdb.Transaction, slide_id: str, patch_count: int, thumbnail_size: int = 0) -> list[Image.Image]:
    patches = []
    for patch_index in range(patch_count):
        patch_id = f"{slide_id}-{patch_index}"
        stored_value = txn.get(patch_id.encode("ascii"))
        if stored_value is None:
            raise KeyError(f"Patch not found: {patch_id}")

        image_bytes = pickle.loads(bytes(stored_value))
        patch = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize if it is larger than 0
        if thumbnail_size > 0:
            patch.thumbnail((thumbnail_size, thumbnail_size))
        patches.append(patch)
    return patches

"""Reconstructs a whole-slide image from patch coordinates and PIL Images"""
def construct_img(coords: np.ndarray, patches_imgs: list[Image.Image], patch_size: int= 512) -> Image.Image:
    if len(coords) == 0 or len(patches_imgs) == 0:
        raise ValueError("coords and patches_imgs must not be empty.")
        
    if len(coords) != len(patches_imgs):
        raise ValueError(f"Mismatch between number of coordinates ({len(coords)}) "
                         f"and patch images ({len(patches_imgs)}).")

    # Get dimensions of loaded thumbnail patch images
    patch_w, patch_h = patches_imgs[0].size
    
    # Check if coords are in level-0 pixel units (e.g., 0, 512, 1024) or grid indices (e.g., 0, 1, 2)
    # Scale coordinates relative to thumbnail patch size
    max_x, max_y = np.max(coords, axis=0)
    
    if max_x > 100 or max_y > 100:  # Coordinates are level-0 pixel values
        scale_x = patch_w / patch_size
        scale_y = patch_h / patch_size
    else:  # Coordinates are grid indices
        scale_x = patch_w
        scale_y = patch_h

    # Calculate overall output canvas size
    canvas_w = int(np.round((max_x * scale_x) + patch_w))
    canvas_h = int(np.round((max_y * scale_y) + patch_h))

    # Create empty white background canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # Paste each patch at its calculated position
    for (x, y), img in zip(coords, patches_imgs):
        pos_x = int(np.round(x * scale_x))
        pos_y = int(np.round(y * scale_y))
        canvas.paste(img, (pos_x, pos_y))

    return canvas

"""Reconstructs WSI with bounding box overlays highlighting patch labels (y_inst == 1)."""
def construct_img_label(coords: np.ndarray,patches_imgs: list[Image.Image],y_inst: np.ndarray,patch_size: int = 512) -> Image.Image:
    """Reconstructs WSI canvas showing ONLY patches with positive instance labels (y_inst == 1)."""
    if len(coords) != len(patches_imgs) or len(coords) != len(y_inst):
        raise ValueError( "coords, patches_imgs, and y_inst must all have the same length.")
    bg_color: tuple = (255, 255, 255)
    patch_w, patch_h = patches_imgs[0].size
    max_x, max_y = np.max(coords, axis=0)

    # Calculate coordinate scaling
    if max_x > 100 or max_y > 100:
        scale_x = patch_w / patch_size
        scale_y = patch_h / patch_size
    else:
        scale_x = patch_w
        scale_y = patch_h

    # Calculate overall canvas bounds
    canvas_w = int(np.round((max_x * scale_x) + patch_w))
    canvas_h = int(np.round((max_y * scale_y) + patch_h))

    # Create a blank background image
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

    # Paste ONLY patches with y_inst == 1
    for (x, y), patch, label in zip(coords, patches_imgs, y_inst):
        if label == 1:
            pos_x = int(np.round(x * scale_x))
            pos_y = int(np.round(y * scale_y))
            canvas.paste(patch, (pos_x, pos_y))

    return canvas

"""Get a sample from a loader"""
def get_sample(loader, slide_id: str):
    for i in loader:
        if slide_id in i['slide_id']:
            return i
    return None

"""Attention heapmap visualization"""
def construct_attn_heatmap(
    coords: np.ndarray,
    attn_weights: np.ndarray,
    patches_imgs: list[Image.Image],
    raw_img: Image.Image,
    patch_size: int = 512,
    cmap_name: str = "jet",
    alpha: float = 0.5
) -> Image.Image:
    # Normalize attention weights to [0, 1]
    attn_weights = np.squeeze(attn_weights).astype(np.float32)
    min_val, max_val = np.min(attn_weights), np.max(attn_weights)
    if max_val - min_val > 1e-8:
        norm_attn = (attn_weights - min_val) / (max_val - min_val)
    else:
        norm_attn = np.zeros_like(attn_weights)

    # Get patch dimensions and scaling factors
    patch_w, patch_h = patches_imgs[0].size
    raw_w, raw_h = raw_img.size
    max_x, max_y = np.max(coords, axis=0)

    if max_x > 100 or max_y > 100:
        scale_x = patch_w / patch_size
        scale_y = patch_h / patch_size
    else:
        scale_x = patch_w
        scale_y = patch_h

    # Build attention map array and tissue mask
    attn_map = np.zeros((raw_h, raw_w), dtype=np.float32)
    tissue_mask = np.zeros((raw_h, raw_w), dtype=bool)

    for (x, y), a in zip(coords, norm_attn):
        pos_x = int(np.round(x * scale_x))
        pos_y = int(np.round(y * scale_y))
        x_end = min(pos_x + patch_w, raw_w)
        y_end = min(pos_y + patch_h, raw_h)

        attn_map[pos_y:y_end, pos_x:x_end] = a
        tissue_mask[pos_y:y_end, pos_x:x_end] = True

    # Colorize attention values
    colormap = plt.get_cmap(cmap_name)
    heatmap_rgba = colormap(attn_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Alpha blend heatmap over raw image ONLY on tissue patches
    raw_arr = np.array(raw_img)
    blended_arr = raw_arr.copy()
    blended_arr[tissue_mask] = (
        (1 - alpha) * raw_arr[tissue_mask] + alpha * heatmap_rgb[tissue_mask]
    ).astype(np.uint8)

    return Image.fromarray(blended_arr)

"""Plot image"""
def plot_img(slide_id, raw_img, lab_img, attn_img, attn_np, gt_lab, pre_lab):
    # Plot side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    fig.suptitle(f"Slide ID: {slide_id}", fontsize=14, y=0.98)
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Raw Image")
    axes[1].imshow(lab_img)
    axes[1].set_title(f"Labeled Image: {gt_lab}")
    axes[2].imshow(attn_img)
    axes[2].set_title(f"Prediction: {pre_lab}")

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)  # Set to False to hide
    for spine in axes[1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)  # Set to False to hide
    for spine in axes[2].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)  # Set to False to hide

    # Add Colorbar for Attention intensity
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=np.min(attn_np), vmax=np.max(attn_np)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.set_title("Attn", fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()

"""Main function"""
def main(args):
    # Load dataset
    dataset = SlideDataset(root_dir= args.dataset_root_dir, dataset_name= args.datasets)
    lmdb_path = f"{args.dataset_root_dir}/{args.datasets}/{args.datasets}.lmdb"
    lmdb_path = lmdb_path if os.path.exists(lmdb_path) else None

    # Initialize data loader for model inference 
    train_loader, val_loader, test_loader = build_dataloader(args= args, image_input= args.image_input)
    print(f"[Dataloader] Train: {len(train_loader)} Val: {len(val_loader)} Test: {len(test_loader)}")

    # Create and define model save directory
    output_base_dir = os.path.join(args.output_path, args.title)
    best_pt_path = os.path.join(output_base_dir, 'model_best.pt')
    # Initialize mil model
    model= build_model(args= args, device= args.device, enc_name= args.enc_name, mil_name= args.mil_name, hf_token= args.hf_token, image_input= args.image_input)
    if args.pretrained_path:
        pretrained_pt_path = args.pretrained_path
    else:
        pretrained_pt_path = best_pt_path
    # Check model path existence
    if not os.path.exists(pretrained_pt_path) or not pretrained_pt_path.endswith(".pt"):
        raise Exception(f"[Pretrained Error] Pretrained model path {pretrained_pt_path} does not exist for validation")
    # Load the saved trained best model
    pretrained_pt = torch.load(pretrained_pt_path, map_location= args.device, weights_only=True)
    model.load_state_dict(pretrained_pt.get('model'))
    print(f"[Model] Loaded best model saved at epoch {pretrained_pt.get('epoch')} from {pretrained_pt_path} for test dataset evaluation")

    # Lmdb is none if not lmdb file found
    if lmdb_path:
        # Load lmdb file
        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,readahead=False, meminit=False)
        with env.begin(write=False, buffers=True) as txn:
            for slide_id, feat, lab, y_inst, coords in tqdm(dataset):
                # Get target sample from dataset loader
                sample = get_sample(loader= train_loader, slide_id= slide_id)
                if sample: # make sure sample is not none
                    # Load patch images
                    patches_imgs = load_slide_patches(txn=txn, slide_id= slide_id, patch_count= len(coords), thumbnail_size= 96)
                    # Construct raw image
                    raw_img = construct_img(coords= coords, patches_imgs= patches_imgs)
                    # Ground-truth image with label
                    lab_img = construct_img_label(coords= coords, patches_imgs= patches_imgs, y_inst= y_inst)
                    # Run model inference
                    logits, attn = model(sample['input'],  return_attn= True)
                    logits = logits[0] if 'dsmil' in args.mil_name else logits # Adjust output format for DS-MIL
                    pre_lab = torch.argmax(logits, dim= 1).item()

                    # Convert attention tensor to 1D NumPy array safely
                    attn_np = attn.detach().cpu().numpy()
                    if attn_np.ndim > 1:
                        # Handle multi-class attention: select attention corresponding to predicted class
                        if attn_np.shape[0] > 1 and attn_np.ndim == 2:
                            attn_np = attn_np[pre_lab]
                        attn_np = np.squeeze(attn_np)
                    # Heatmap image
                    attn_img = construct_attn_heatmap(
                        coords= coords,attn_weights= attn_np,patches_imgs= patches_imgs,raw_img= raw_img)
                    # Plot side-by-side comparison
                    plot_img(slide_id= slide_id, raw_img= raw_img,lab_img= lab_img,attn_img= attn_img,attn_np= attn_np,gt_lab= lab,pre_lab= pre_lab)
                    
if __name__ == "__main__":
    args = get_parse_args()
    main(args= args)