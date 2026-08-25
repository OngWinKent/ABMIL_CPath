import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from options import get_parse_args
import lmdb
import pickle
from PIL import Image, ImageDraw
import io
from datasets import build_dataloader
from modules import build_model
import matplotlib.pyplot as plt
from datasets import data_utils
import cv2

"""Plot image"""
def plot_img(slide_id, raw_img, lab_img, attn_img, attn_np, gt_lab, pre_lab):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    fig.suptitle(f"Slide ID: {slide_id}", fontsize=14, y=0.98)
    
    axes[0].imshow(raw_img)
    axes[0].set_title("Raw Image")
    
    axes[1].imshow(lab_img)
    axes[1].set_title(f"Labeled Image: {gt_lab}")
    
    axes[2].imshow(attn_img)
    axes[2].set_title(f"Pred: {pre_lab}") 

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)
    for spine in axes[1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)
    for spine in axes[2].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)

    # Add Colorbar for Attention intensity
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=np.min(attn_np), vmax=np.max(attn_np)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.set_title("Attn", fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()

# Lmdb visualization
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

"""Reconstructs WSI showing ONLY patches with positive instance labels (y_inst == 1)."""
def construct_img_label(
    coords: np.ndarray,
    patches_imgs: list[Image.Image],
    y_inst: np.ndarray,
    patch_size: int = 512,
    contour_color: tuple = (255, 0, 0),  # RGB format (Red)
    contour_thickness: int = 4,
) -> Image.Image:
  """Reconstructs WSI showing ALL patches, drawing a unified outer boundary around contiguous regions with label == 1."""
  if len(coords) != len(patches_imgs) or len(coords) != len(y_inst):
    raise ValueError(
        "coords, patches_imgs, and y_inst must all have the same length."
    )

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

  # Create blank canvas and binary mask for positive regions
  canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
  mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

  # Step 1: Paste all patches and build the binary mask for label == 1
  for (x, y), patch, label in zip(coords, patches_imgs, y_inst):
    pos_x = int(np.round(x * scale_x))
    pos_y = int(np.round(y * scale_y))

    # Paste every patch onto the canvas
    canvas.paste(patch, (pos_x, pos_y))

    # Mark positive regions on the mask
    if label == 1:
      mask[pos_y : pos_y + patch_h, pos_x : pos_x + patch_w] = 255

  # Extract external contours using OpenCV
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw outer boundaries on the final image
  canvas_np = np.array(canvas)

  # Convert RGB to BGR for OpenCV drawing (OpenCV works in BGR format)
  cv2.drawContours(canvas_np, contours, -1, contour_color, thickness=contour_thickness)

  return Image.fromarray(canvas_np)

"""Attention heatmap visualization"""
def construct_attn_heatmap(
    coords: np.ndarray,
    attn_weights: np.ndarray,
    patches_imgs: list[Image.Image],
    raw_img: Image.Image,
    is_clip_weights: bool= True,
    patch_size: int = 512,
    cmap_name: str = "jet",
    alpha: float = 0.5
) -> Image.Image:
    # Normalize attention weights to [0, 1]
    attn_weights = np.squeeze(attn_weights).astype(np.float32)

    # Clip weight for better visualization 
    if is_clip_weights:
        min_val = np.min(attn_weights)
        p_max = np.percentile(attn_weights, 99.5) # Clip the top 0.5% extreme outliers so the colormap isn't squashed by a single patch
        attn_clipped = np.clip(attn_weights, min_val, p_max)
        if p_max - min_val > 1e-8:
            norm_attn = (attn_clipped - min_val) / (p_max - min_val)
            norm_attn = np.power(norm_attn, 0.2)  # Gamma = 0.2 pulls low-tier attention values up into the visible range (cyan/yellow)
        else:
            norm_attn = np.zeros_like(attn_weights)
    # Raw weight visualization
    else:
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
    blended_arr[tissue_mask] = ((1 - alpha) * raw_arr[tissue_mask] + alpha * heatmap_rgb[tissue_mask]).astype(np.uint8)

    return Image.fromarray(blended_arr)

# ---------------- Dot visualization ---------
"""Helper to render dots onto a PIL Image canvas based on coordinates."""
def _render_dot_canvas(
    coords: np.ndarray,
    point_colors: list = None,
    default_color: tuple = (180, 180, 180),
    patch_size: int = 512,
    scale_factor: int = 12,
    dot_radius: int = 4,
) -> Image.Image:
    if len(coords) == 0:
        return Image.new("RGB", (100, 100), color=(255, 255, 255))

    max_x, max_y = np.max(coords, axis=0)
    if max_x > 100 or max_y > 100:  # Pixel coordinates
        gx = np.round(coords[:, 0] / patch_size).astype(int)
        gy = np.round(coords[:, 1] / patch_size).astype(int)
    else:  # Grid coordinates
        gx = np.round(coords[:, 0]).astype(int)
        gy = np.round(coords[:, 1]).astype(int)

    max_gx, max_gy = np.max(gx), np.max(gy)
    canvas_w = int((max_gx + 2) * scale_factor)
    canvas_h = int((max_gy + 2) * scale_factor)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (x, y) in enumerate(zip(gx, gy)):
        cx = (x + 1) * scale_factor
        cy = (y + 1) * scale_factor
        # Safe lookup: falls back to default_color if index exceeds point_colors
        if point_colors is not None and i < len(point_colors):
            color = point_colors[i]
        else:
            color = default_color
        draw.ellipse(
            [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
            fill=color,
        )

    return canvas

"""Helper to render dots onto a PIL Image canvas with variable sizes and z-ordering."""
def _render_dot_canvas_variable(
    coords: np.ndarray,
    point_colors: list = None,
    default_color: tuple = (180, 180, 180),
    dot_radii: list = None,
    patch_size: int = 512,
    scale_factor: int = 12,
    default_dot_radius: int = 4,
) -> Image.Image:
    if len(coords) == 0:
        return Image.new("RGB", (100, 100), color=(255, 255, 255))

    max_x, max_y = np.max(coords, axis=0)
    if max_x > 100 or max_y > 100:  # Pixel coordinates
        gx = np.round(coords[:, 0] / patch_size).astype(int)
        gy = np.round(coords[:, 1] / patch_size).astype(int)
    else:  # Grid coordinates
        gx = np.round(coords[:, 0]).astype(int)
        gy = np.round(coords[:, 1]).astype(int)

    max_gx, max_gy = np.max(gx), np.max(gy)
    canvas_w = int((max_gx + 2) * scale_factor)
    canvas_h = int((max_gy + 2) * scale_factor)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Sort indices so larger/important dots are drawn LAST (on top)
    indices = list(range(len(coords)))
    if dot_radii is not None:
        indices.sort(key=lambda idx: dot_radii[idx])

    for i in indices:
        cx = (gx[i] + 1) * scale_factor
        cy = (gy[i] + 1) * scale_factor
        
        color = point_colors[i] if (point_colors is not None and i < len(point_colors)) else default_color
        r = dot_radii[i] if (dot_radii is not None and i < len(dot_radii)) else default_dot_radius

        draw.ellipse( [cx - r, cy - r, cx + r, cy + r],fill=color)

    return canvas

"""Raw image fallback: neutral gray dots."""
def construct_dot_raw_img(
    coords: np.ndarray, patch_size: int = 512
) -> Image.Image:
    
    return _render_dot_canvas(coords, default_color=(128, 128, 128), patch_size=patch_size)

"""Label image fallback: gray dots for 0, RED dots for 1."""
def construct_dot_lab_img(
    coords: np.ndarray, y_inst: np.ndarray, patch_size: int = 512
) -> Image.Image:
    y_inst = np.asarray(y_inst).reshape(-1)

    # Match length with coords
    if len(y_inst) < len(coords):
        y_inst = np.pad(y_inst, (0, len(coords) - len(y_inst)), mode="constant")
    elif len(y_inst) > len(coords):
        y_inst = y_inst[: len(coords)]

    colors = [(255, 0, 0) if y == 1 else (180, 180, 180) for y in y_inst]
    return _render_dot_canvas(coords, point_colors=colors, patch_size=patch_size)

"""Heatmap fallback: Aggressive scaling and percentile clipping for sparse MIL attention."""
def construct_dot_attn_heatmap(
    coords: np.ndarray,
    attn_weights: np.ndarray,
    patch_size: int = 512,
    cmap_name: str = "jet",
) -> Image.Image:
    attn_weights = np.asarray(attn_weights, dtype=np.float32).reshape(-1)

    # Align length with coords
    if len(attn_weights) < len(coords):
        attn_weights = np.pad(attn_weights, (0, len(coords) - len(attn_weights)), mode="constant")
    elif len(attn_weights) > len(coords):
        attn_weights = attn_weights[: len(coords)]

    # Clip the top 0.5% extreme outliers so the colormap isn't squashed by a single patch
    min_val = np.min(attn_weights)
    p_max = np.percentile(attn_weights, 99.5) 
    attn_clipped = np.clip(attn_weights, min_val, p_max)
    if p_max - min_val > 1e-8:
        norm_attn = (attn_clipped - min_val) / (p_max - min_val)
        # Gamma = 0.2 pulls low-tier attention values up into the visible range (cyan/yellow)
        norm_attn = np.power(norm_attn, 0.2)  
    else:
        norm_attn = np.zeros_like(attn_weights)

    colormap = plt.get_cmap(cmap_name)
    rgba_colors = colormap(norm_attn)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in rgba_colors]

    # Background dots = 2px (less cluttered), Top attention dots = up to 14px (highly visible)
    radii = [int(2 + 12 * val) for val in norm_attn]

    return _render_dot_canvas_variable(
        coords=coords, 
        point_colors=colors, 
        dot_radii=radii, 
        patch_size=patch_size,
        default_dot_radius=2
    )

"""Main function"""
def main(args):
    # Load dataset
    dataset = data_utils.SlideDataset(root_dir= args.dataset_root_dir, dataset_name= args.datasets)

    # Path configuration
    lmdb_path = f"{args.dataset_root_dir}/{args.datasets}/{args.datasets}.lmdb"
    lmdb_path = lmdb_path if os.path.exists(lmdb_path) else None
    raw_img_folder = f"{args.dataset_root_dir}/{args.datasets}/raw_imgs"
    raw_img_folder = raw_img_folder if os.path.isdir(raw_img_folder) else None

    # Initialize data loader for model inference 
    train_loader, val_loader, test_loader = build_dataloader(args= args, image_input= args.image_input, inference= True)
    print(f"[Dataloader] Train: {len(train_loader)} Val: {len(val_loader)} Test: {len(test_loader)}")

    # Create and define model save directory
    output_base_dir = os.path.join(args.output_path, args.title)
    best_pt_path = os.path.join(output_base_dir, 'model_best.pt')
    
    # Initialize mil model
    model = build_model(args= args, device= args.device, enc_name= args.enc_name, mil_name= args.mil_name, hf_token= args.hf_token, image_input= args.image_input)
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

    # Open LMDB environment if available
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False) if lmdb_path else None

    try:
        for slide_id, feat, lab, y_inst, coords in tqdm(dataset):
            # Bypass dataloader for perfect coordinate alignment
            feat = feat.to(args.device)
            if feat.ndim == 2:
                feat = feat.unsqueeze(0) 
                
            logits, attn = model(feat, return_attn=True)
            logits = logits[0] if 'dsmil' in args.mil_name else logits
            pre_lab = torch.argmax(logits, dim=1).item()
            gt_idx = lab.item()

            # Multi-class attention distribution
            attn_np = attn.detach().cpu().numpy()
            attn_np = np.squeeze(attn_np) # Flatten attention weights
            
            # If the array is 2D, we have multi-branch attention
            if attn_np.ndim == 2:
                # Verify the output matches your expected number of classes
                if attn_np.shape[0] == args.n_classes:
                    # Select the attention branch corresponding to the GROUND-TRUTH class
                    branch_idx = min(gt_idx, args.n_classes - 1)
                    attn_np = attn_np[branch_idx]
                else:
                    # Fallback for unexpected shapes (e.g., intermediate tensor outputs)
                    attn_np = attn_np[0]
            
            # Generate image plots
            if env is not None: # Draw with encoded image from lmdb
                with env.begin(write=False, buffers=True) as txn:
                    patches_imgs = load_slide_patches(txn=txn, slide_id=slide_id, patch_count=len(coords), thumbnail_size=96) # raw patches images in tiles
                    raw_img = construct_img(coords=coords, patches_imgs=patches_imgs)
                    lab_img = construct_img_label(coords=coords, patches_imgs=patches_imgs, y_inst=y_inst, contour_thickness= 15, contour_color= (0, 255, 0))
                    attn_img = construct_attn_heatmap(coords=coords, attn_weights=attn_np, patches_imgs=patches_imgs, raw_img=raw_img, is_clip_weights= True)
            
            # Dot plots
            else: 
                raw_img = construct_dot_raw_img(coords=coords)
                lab_img = construct_dot_lab_img(coords=coords, y_inst=y_inst)
                attn_img = construct_dot_attn_heatmap(coords=coords, attn_weights=attn_np)

            # Plot side-by-side comparison
            plot_img(slide_id=slide_id, raw_img=raw_img, lab_img=lab_img, attn_img=attn_img, attn_np=attn_np, gt_lab=lab, pre_lab=pre_lab)
            
    finally:
        if env is not None:
            env.close()  

if __name__ == "__main__":
    args = get_parse_args()
    main(args= args)