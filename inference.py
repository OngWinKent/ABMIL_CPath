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
from modules import build_model
import matplotlib.pyplot as plt
from datasets import data_utils
import cv2
from typing import *
import copy

"""Main attention weights heatmap inference class"""
class InferenceAttn:
    def __init__(self, args):
        # Global variables
        self.device = args.device
        self.n_classes = args.n_classes
        self.mil_name = args.mil_name

        # Open LMDB environment if available
        lmdb_path = f"{args.dataset_root_dir}/{args.datasets}/{args.datasets}.lmdb"
        if lmdb_path and os.path.exists(lmdb_path):
            self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        else:
            raise Exception(f".lmdb path not exist {lmdb_path}")

        # Load dataset
        self.dataset = data_utils.SlideDataset(root_dir= args.dataset_root_dir, dataset_name= args.datasets)

        # Create and define model save directory
        output_base_dir = os.path.join(args.output_path, args.title)
        best_pt_path = os.path.join(output_base_dir, 'model_best.pt')
        # Initialize mil model
        self.model = build_model(args= args, device= args.device, enc_name= args.enc_name, mil_name= args.mil_name, hf_token= args.hf_token, image_input= args.image_input)
        if args.pretrained_path:
            pretrained_pt_path = args.pretrained_path
        else:
            pretrained_pt_path = best_pt_path
        # Check model path existence
        if not os.path.exists(pretrained_pt_path) or not pretrained_pt_path.endswith(".pt"):
            raise Exception(f"[Pretrained Error] Pretrained model path {pretrained_pt_path} does not exist for validation")
        # Load the saved trained best model
        pretrained_pt = torch.load(pretrained_pt_path, map_location= args.device, weights_only=True)
        self.model.load_state_dict(pretrained_pt.get('model'))
        print(f"[Model] Loaded best model saved at epoch {pretrained_pt.get('epoch')} from {pretrained_pt_path} for test dataset evaluation")

    """Load every patch belonging to one slide from LMDB."""
    def _load_slide_patches(self, txn: lmdb.Transaction, slide_id: str, patch_count: int, thumbnail_size: int = 0) -> list[Image.Image]:
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

    """Reconstructs WSI showing ONLY patches with positive instance labels (y_inst == 1)."""
    def _construct_img_label(self, coords: np.ndarray,patches_imgs: list[Image.Image],y_inst: np.ndarray,contour_color: tuple, patch_size: int = 512) -> Tuple[Image.Image, Image.Image]:
        if len(coords) != len(patches_imgs) or len(coords) != len(y_inst):
            raise ValueError( "coords, patches_imgs, and y_inst must all have the same length." )

        min_contour_thickness = 10
        max_contour_thickness = 80
        bg_color: tuple = (255, 255, 255) # White color
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
        raw_img = np.array(canvas)
        labelled_img = copy.deepcopy(raw_img)

        # Dynamic contour thickness computation based on pixel
        contour_thickness = max(min_contour_thickness, min(int(canvas_h*canvas_w / 1_000_000), max_contour_thickness))

        # Convert RGB to BGR for OpenCV drawing (OpenCV works in BGR format)
        cv2.drawContours(labelled_img, contours, -1, contour_color, thickness=contour_thickness)

        return Image.fromarray(raw_img), Image.fromarray(labelled_img)

    """Attention heatmap visualization"""
    def _construct_attn_heatmap(self, coords: np.ndarray,attn_weights: np.ndarray,patches_imgs: list[Image.Image],raw_img: Image.Image,
        is_clip_weights: bool= True, patch_size: int = 512,cmap_name: str = "jet",alpha: float = 0.5) -> Image.Image:
        # Normalize attention weights to [0, 1]
        attn_weights = np.squeeze(attn_weights).astype(np.float32)

        # Clip weight for better visualization 
        if is_clip_weights:
            min_val = np.min(attn_weights)
            # Adjust based on visualization
            #max_val = np.percentile(attn_weights, 99.5) # Clip the top 0.5% extreme outliers so the colormap isn't squashed by a single patch
            max_val = np.percentile(attn_weights, 99) # Clip the top 1% extreme outliers so the colormap isn't squashed by a single patch
            attn_weights = np.clip(attn_weights, min_val, max_val)
        # Raw weight visualization
        else:
            min_val, max_val = np.min(attn_weights), np.max(attn_weights)
        # Normalize attention weights
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

    """Plot attention heatmaps images"""
    def _plot_img(self, slide_id: int, raw_img: Image.Image, lab_img: Image.Image, attn_img: Image.Image, attn_np: np.ndarray, gt_lab: int, pre_lab: int) -> None:
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

    """Main function"""
    def __call__(self):
        try:
            for slide_id, feat, lab, y_inst, coords in tqdm(self.dataset):

                feat = feat.to(self.device)
                if feat.ndim == 2:
                    feat = feat.unsqueeze(0) 
                
                logits, attn = self.model(feat, return_attn=True)
                logits = logits[0] if 'dsmil' in self.mil_name else logits
                pre_lab = torch.argmax(logits, dim=1).item()
                gt_idx = lab.item()

                # Multi-class attention distribution
                attn_np = attn.detach().cpu().numpy()
                # If the array is 2D, we have multi-branch attention
                if attn_np.ndim == 2:
                    # Verify the output matches your expected number of classes
                    if attn_np.shape[0] == self.n_classes:
                        # Select the attention branch corresponding to the GROUND-TRUTH class
                        branch_idx = min(gt_idx, self.n_classes - 1)
                        attn_np = attn_np[branch_idx]
                    else:
                        # Fallback for unexpected shapes (e.g., intermediate tensor outputs)
                        attn_np = attn_np[0]
                attn_np = np.squeeze(attn_np) # Flatten attention weights

                # Generate image plots
                with self.env.begin(write=False, buffers=True) as txn:
                    patches_imgs = self._load_slide_patches(txn=txn, slide_id=slide_id, patch_count=len(coords), thumbnail_size=96) # raw patches images in tiles
                    raw_img, lab_img = self._construct_img_label(coords=coords, patches_imgs=patches_imgs, y_inst=y_inst, contour_color= (0, 255, 0))
                    attn_img = self._construct_attn_heatmap(coords=coords, attn_weights=attn_np, patches_imgs=patches_imgs, raw_img=raw_img, is_clip_weights= True)

                # Plot side-by-side comparison
                self._plot_img(slide_id=slide_id, raw_img=raw_img, lab_img=lab_img, attn_img=attn_img, attn_np=attn_np, gt_lab=lab, pre_lab=pre_lab)
                
        finally:
            if self.env is not None:
                self.env.close()  

if __name__ == "__main__":
    # Get parse arguments
    args = get_parse_args()
    inf_attn = InferenceAttn(args)
    inf_attn()