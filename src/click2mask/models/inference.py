from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import io
import numpy as np
from PIL import Image


class MaskGenerator:
    def __init__(self):
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint = "/opt/packages/sam2/checkpoints/sam2.1_hiera_large.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_model = build_sam2(
            model_cfg,
            checkpoint,
            device=device,
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=64,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.88,
            crop_n_layers=0,
            # min_mask_region_area=100,
            # crop_n_points_downscale_factor=2,
            # point_grids=None,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Turn image into RGBA format (for png) and apply mask"""

        # if image is RGB, add alpha channel
        if image.shape[2] == 3:
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = 255
            image = rgba

        mask = mask.astype(np.bool)
        image[~mask, 3] = 0  # set alpha channel to 0 for all pixels not in mask
        return image

    def process_point_mask(
        self, image: np.ndarray, point_x: int, point_y: int
    ) -> bytes:
        """Process image with single input point and return masked image"""
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([[point_x, point_y]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

        # Convert to PIL image and save as PNG
        masked_image = self.apply_mask_to_image(image, masks[0])
        img = Image.fromarray(masked_image, "RGBA")
        img_byte_array = io.BytesIO() 
        img.save(img_byte_array, format="PNG") 
        return img_byte_array.getvalue()
