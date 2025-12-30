"""
Background Removal Utilities
============================

Functions for removing backgrounds using segmentation masks.
Supports two-stage approach: YOLO detection + SAM segmentation.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import warnings


def detect_leaves_with_yolo(
    image_path: str,
    conf_threshold: float = 0.10
) -> Optional[List[List[float]]]:
    """
    Detect all leaves in an image using YOLO.
    
    Note: YOLOv8 pretrained on COCO doesn't have a 'leaf' class. This function
    uses a very low confidence threshold to detect any objects that might be leaves.
    For production use, fine-tune YOLO on your soybean leaf dataset.
    
    Args:
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections (default: 0.10 - very permissive)
        
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...] or None if no detections
        
    Example:
        >>> bboxes = detect_leaves_with_yolo('leaf.jpg')
        >>> print(f"Found {len(bboxes)} leaves")
    """
    try:
        from ultralytics import YOLO
        from PIL import Image
        
        # Use YOLOv8 nano for fast detection
        # Note: COCO dataset doesn't have 'leaf' class, so we use low threshold
        # For best results, fine-tune YOLO on your leaf dataset
        model = YOLO('yolov8n.pt')
        
        # Run detection with low confidence to catch leaves
        results = model(image_path, conf=conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        
        # Extract bounding boxes
        # Since COCO doesn't have 'leaf', we accept all detections above threshold
        # This may include false positives, but SAM will refine the segmentation
        bboxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()
            
            # Get image dimensions to filter out full-image detections
            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h
            
            for box, conf in zip(boxes, confs):
                if conf >= conf_threshold:
                    # Filter out boxes that cover almost the entire image
                    x1, y1, x2, y2 = box
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    
                    # Only accept boxes that cover 5%-95% of image
                    # This filters out full-image and tiny detections
                    if 0.05 * img_area < box_area < 0.95 * img_area:
                        bboxes.append(box.tolist())
        
        return bboxes if len(bboxes) > 0 else None
        
    except Exception as e:
        warnings.warn(f"YOLO detection failed: {e}. Falling back to center point.")
        return None


def fix_mask_orientation(
    mask: np.ndarray,
    expected_height: int,
    expected_width: int
) -> Optional[np.ndarray]:
    """
    Fix mask orientation to match expected image dimensions.
    
    Args:
        mask: Binary mask that may be transposed
        expected_height: Expected image height
        expected_width: Expected image width
        
    Returns:
        Corrected mask or None if unfixable
    """
    mask_bool = mask.astype(bool)
    
    # If already correct, return as-is
    if mask_bool.shape == (expected_height, expected_width):
        return mask_bool
    
    # If transposed, fix it
    if mask_bool.shape == (expected_width, expected_height):
        return mask_bool.T
    
    # Unfixable shape mismatch
    warnings.warn(f"Cannot fix mask shape {mask_bool.shape} to match ({expected_height}, {expected_width})")
    return None


def is_likely_background_mask(
    mask: np.ndarray,
    image_height: int,
    image_width: int,
    edge_threshold: float = 0.7
) -> bool:
    """
    Check if a mask is likely background by testing if it heavily touches image edges.
    
    A mask is considered background if it touches multiple edges heavily OR
    if it covers an entire edge (likely to be the image border itself).
    
    Args:
        mask: Binary mask
        image_height: Image height
        image_width: Image width
        edge_threshold: Fraction of edge that must be covered (default: 70%)
        
    Returns:
        True if mask is likely background
    """
    # Check all four edges
    top_edge = np.sum(mask[0, :]) / image_width
    bottom_edge = np.sum(mask[-1, :]) / image_width
    left_edge = np.sum(mask[:, 0]) / image_height
    right_edge = np.sum(mask[:, -1]) / image_height
    
    # Count how many edges are heavily touched (>70%)
    heavy_edges = sum([
        top_edge > edge_threshold,
        bottom_edge > edge_threshold,
        left_edge > edge_threshold,
        right_edge > edge_threshold
    ])
    
    # Background if:
    # 1. Any single edge is almost completely covered (>90%)
    # 2. Two or more edges are heavily touched (>70%)
    if any([top_edge > 0.9, bottom_edge > 0.9, left_edge > 0.9, right_edge > 0.9]):
        return True
    
    if heavy_edges >= 2:
        return True
    
    return False


def remove_background(
    image: Image.Image,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Remove background by applying a binary mask.
    
    Args:
        image: Original PIL Image (RGB)
        mask: Binary mask (H, W) - True for foreground, False for background
        background_color: RGB color for background pixels (default: black)
        
    Returns:
        PIL Image with background removed
        
    Example:
        >>> cleaned_image = remove_background(image, leaf_mask, background_color=(0, 0, 0))
    """
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)
    
    # Check dimensions match
    if image_np.shape[:2] != mask_bool.shape:
        raise ValueError(
            f"Image shape {image_np.shape[:2]} doesn't match mask shape {mask_bool.shape}"
        )
    
    # Create output image
    output = image_np.copy()
    
    # Set background pixels to specified color
    output[~mask_bool] = background_color
    
    # Convert back to PIL Image
    return Image.fromarray(output)


def remove_background_with_alpha(
    image: Image.Image,
    mask: np.ndarray
) -> Image.Image:
    """
    Remove background by creating an RGBA image with transparency.
    
    Args:
        image: Original PIL Image (RGB)
        mask: Binary mask (H, W)
        
    Returns:
        PIL Image (RGBA) with transparent background
        
    Example:
        >>> transparent_image = remove_background_with_alpha(image, leaf_mask)
    """
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)
    
    # Create RGBA array
    output = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
    output[:, :, :3] = image_np  # Copy RGB channels
    output[:, :, 3] = mask_bool.astype(np.uint8) * 255  # Alpha channel
    
    # Convert to PIL Image
    return Image.fromarray(output, mode='RGBA')


def process_image_with_sam(
    image_path: str,
    model,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    text_prompt: Optional[str] = None,
    point_prompt: Optional[list] = None,
    bbox_prompt: Optional[list] = None,
    merge_threshold: float = 0.02,
    use_yolo_detection: bool = True
) -> Tuple[Image.Image, np.ndarray, dict]:
    """
    Complete pipeline: segment image with SAM and remove background.
    
    **Multi-Point Grid Approach** (default):
    1. Creates a 5x5 grid of sample points across the image
    2. SAM generates masks from each point
    3. Intelligently merges masks with deduplication (removes 80%+ overlaps)
    4. Results in complete leaf coverage without missing parts
    
    Alternative prompts:
    - Text prompt: For SAM3 semantic segmentation (e.g., 'leaf', 'plant')
    - Point prompt: SAM2-style point prompts [[x, y]], labels=[1]
    - BBox prompt: Manual bounding boxes [[x1, y1, x2, y2]]
    
    Args:
        image_path: Path to input image
        model: Ultralytics SAM model
        background_color: Background color for removed pixels
        text_prompt: Text description for SAM3 (e.g., 'leaf')
        point_prompt: Point coordinates [[x, y]] with labels=[1]
        bbox_prompt: Manual bounding boxes [[x1, y1, x2, y2]]
        merge_threshold: Merge all masks above this fraction of image area (default: 2%)
        use_yolo_detection: Use multi-point grid strategy (default: True, recommended)
        
    Returns:
        Tuple of (cleaned_image, mask_used, metadata)
        
    Example:
        >>> model = load_sam_model('sam_b.pt')
        >>> # Multi-point grid (recommended for complete coverage)
        >>> cleaned, mask, meta = process_image_with_sam('leaf.jpg', model)
        >>> print(f"Found {meta['num_masks_found']} masks, merged {meta['num_masks_merged']}")
    """
    from .sam_utils import segment_image, get_all_masks, get_mask_statistics, segment_image_with_text
    
    # Load image for processing
    image = Image.open(image_path)
    image_area = image.width * image.height
    min_mask_area = int(image_area * merge_threshold)
    
    # MULTI-POINT APPROACH: Use grid of points for better coverage
    if use_yolo_detection and text_prompt is None and point_prompt is None and bbox_prompt is None:
        print(f"  Using multi-point grid strategy for complete coverage...")
        
        # Create a grid of points across the image to catch all leaves
        # This is much more reliable than YOLO which doesn't know what leaves are
        grid_size = 5  # 5x5 grid = 25 points
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int((j + 0.5) * image.width / grid_size)
                y = int((i + 0.5) * image.height / grid_size)
                points.append([x, y])
        
        # Generate masks from all grid points
        all_masks = []
        print(f"  Generating masks from {len(points)} sample points...")
        
        # Process in batches to avoid overwhelming SAM
        batch_size = 5
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            labels = [1] * len(batch_points)  # All foreground
            
            results = segment_image(image_path, model, points=batch_points, labels=labels)
            masks = get_all_masks(results)
            if masks is not None:
                all_masks.extend(masks)
        
        if len(all_masks) == 0:
            warnings.warn("SAM found no masks from grid points. Using center point.")
            point_prompt = [[image.width // 2, image.height // 2]]
            labels = [1]
            results = segment_image(image_path, model, points=point_prompt, labels=labels)
            all_masks = get_all_masks(results)
            
            if all_masks is None:
                full_mask = np.ones((image.height, image.width), dtype=bool)
                metadata = {
                    'num_masks_found': 0,
                    'num_masks_merged': 0,
                    'mask_area': 0,
                    'mask_coverage': 0.0
                }
                return image, full_mask, metadata
            
        
        # Merge all leaf masks with intelligent deduplication
        final_mask = np.zeros((image.height, image.width), dtype=bool)
        num_masks_used = 0
        skipped_background = 0
        
        # Sort masks by area (largest first) for better merging
        mask_areas = [(i, np.sum(mask.astype(bool))) for i, mask in enumerate(all_masks)]
        mask_areas.sort(key=lambda x: x[1], reverse=True)
        
        for idx, area in mask_areas:
            mask = all_masks[idx]
            
            # Fix mask orientation if needed
            mask_bool = fix_mask_orientation(mask, image.height, image.width)
            if mask_bool is None:
                continue
            
            # Skip masks that are likely background (heavily touch multiple edges)
            if is_likely_background_mask(mask_bool, image.height, image.width):
                skipped_background += 1
                continue
            
            # Include masks above threshold
            # Also check if this mask significantly overlaps with what we already have
            if area >= min_mask_area:
                # Calculate overlap with existing mask
                overlap = np.sum(final_mask & mask_bool)
                overlap_ratio = overlap / area if area > 0 else 1.0
                
                # Only add if it's not mostly redundant (less than 60% overlap)
                # Lowered from 80% to keep more leaf parts
                if overlap_ratio < 0.6:
                    final_mask = final_mask | mask_bool
                    num_masks_used += 1
        
        # If no masks passed threshold, merge the largest non-background masks
        if num_masks_used == 0:
            # Use top 5 largest non-background masks
            for idx, area in mask_areas[:10]:  # Check more candidates
                mask = all_masks[idx]
                
                # Fix mask orientation
                mask_bool = fix_mask_orientation(mask, image.height, image.width)
                if mask_bool is None:
                    continue
                
                # Skip background masks
                if is_likely_background_mask(mask_bool, image.height, image.width):
                    continue
                
                final_mask = final_mask | mask_bool
                num_masks_used += 1
                
                if num_masks_used >= 5:  # Limit to 5 masks
                    break
        
        # Apply morphological closing to fill small holes and complete leaf shapes
        # This helps keep leaves intact even if some parts were missed
        import cv2
        kernel_size = max(5, int(min(image.height, image.width) * 0.01))  # 1% of image size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        
        # Remove background
        cleaned_image = remove_background(image, final_mask, background_color)
        
        print(f"  Merged {num_masks_used} masks, skipped {skipped_background} background masks")
        
        # Metadata
        mask_area = int(np.sum(final_mask))
        metadata = {
            'num_masks_found': len(all_masks),
            'num_masks_merged': num_masks_used,
            'mask_area': mask_area,
            'mask_coverage': float(mask_area / image_area),
            'method': 'multi_point_grid'
        }
        
        return cleaned_image, final_mask, metadata
    
    # FALLBACK: Original methods (text, point, bbox prompts)
    # Generate segmentation based on prompt type
    if text_prompt:
        # Use SAM3 semantic predictor for text
        masks, boxes = segment_image_with_text(image_path, [text_prompt], model_path=model if isinstance(model, str) else 'sam3.pt')
        if masks is None:
            warnings.warn(f"No masks found for text prompt: {text_prompt}")
            full_mask = np.ones((image.height, image.width), dtype=bool)
            metadata = {'num_masks_found': 0, 'mask_area': 0, 'mask_coverage': 0.0}
            return image, full_mask, metadata
        
        num_masks = len(masks)
        all_masks = masks
        
    else:
        # Use point or bbox prompt (or default center point)
        if point_prompt is None and bbox_prompt is None:
            # Default: use center point
            point_prompt = [[image.width // 2, image.height // 2]]
        
        labels = [1] if point_prompt else None
        results = segment_image(image_path, model, points=point_prompt, labels=labels, bboxes=bbox_prompt)
        
        # Extract ALL masks
        all_masks = get_all_masks(results)
        if all_masks is None:
            warnings.warn("No masks found with provided prompts")
            full_mask = np.ones((image.height, image.width), dtype=bool)
            metadata = {'num_masks_found': 0, 'mask_area': 0, 'mask_coverage': 0.0}
            return image, full_mask, metadata
        
        stats = get_mask_statistics(results)
        num_masks = stats['num_masks']
    
    # Filter and merge masks above threshold
    final_mask = np.zeros((image.height, image.width), dtype=bool)
    num_masks_used = 0
    
    for mask in all_masks:
        # Fix mask orientation if needed
        mask_bool = fix_mask_orientation(mask, image.height, image.width)
        if mask_bool is None:
            continue
        
        # Skip likely background masks
        if is_likely_background_mask(mask_bool, image.height, image.width):
            continue
        
        mask_area = np.sum(mask_bool)
        
        # Include masks above threshold
        if mask_area >= min_mask_area:
            final_mask = final_mask | mask_bool
            num_masks_used += 1
    
    # If no masks passed threshold, use the largest non-background mask
    if num_masks_used == 0:
        # Find largest non-background mask
        for mask in sorted(all_masks, key=lambda m: np.sum(m), reverse=True):
            mask_bool = fix_mask_orientation(mask, image.height, image.width)
            if mask_bool is None:
                continue
            if not is_likely_background_mask(mask_bool, image.height, image.width):
                final_mask = mask_bool
                num_masks_used = 1
                break
        
        # If all masks are background, use the largest one anyway
        if num_masks_used == 0:
            for mask in sorted(all_masks, key=lambda m: np.sum(m), reverse=True):
                mask_bool = fix_mask_orientation(mask, image.height, image.width)
                if mask_bool is not None:
                    final_mask = mask_bool
                    num_masks_used = 1
                    break
    
    # Apply morphological closing to complete leaves
    import cv2
    kernel_size = max(5, int(min(image.height, image.width) * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    
    # Remove background
    cleaned_image = remove_background(image, final_mask, background_color)
    
    # Metadata
    mask_area = int(np.sum(final_mask))
    metadata = {
        'num_masks_found': num_masks,
        'num_masks_merged': num_masks_used,
        'mask_area': mask_area,
        'mask_coverage': float(mask_area / image_area),
    }
    
    return cleaned_image, final_mask, metadata


def apply_morphological_operations(
    mask: np.ndarray,
    operation: str = 'close',
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask
        operation: 'close', 'open', 'erode', 'dilate'
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned mask
        
    Example:
        >>> cleaned_mask = apply_morphological_operations(mask, 'close', kernel_size=5)
    """
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV not installed, skipping morphological operations")
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = mask.astype(np.uint8) * 255
    
    if operation == 'close':
        result = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        result = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    elif operation == 'erode':
        result = cv2.erode(mask_uint8, kernel)
    elif operation == 'dilate':
        result = cv2.dilate(mask_uint8, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return (result > 127).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask with potential holes
        
    Returns:
        Mask with holes filled
        
    Example:
        >>> filled_mask = fill_holes(mask)
    """
    try:
        from scipy import ndimage
    except ImportError:
        warnings.warn("SciPy not installed, skipping hole filling")
        return mask
    
    # Fill holes
    filled = ndimage.binary_fill_holes(mask)
    
    return filled.astype(bool)


def smooth_mask_edges(
    mask: np.ndarray,
    sigma: float = 2.0,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Smooth mask edges using Gaussian blur.
    
    Args:
        mask: Binary mask
        sigma: Gaussian blur sigma
        threshold: Threshold for binarization after blur
        
    Returns:
        Smoothed mask
        
    Example:
        >>> smooth_mask = smooth_mask_edges(mask, sigma=2.0)
    """
    try:
        from scipy import ndimage
    except ImportError:
        warnings.warn("SciPy not installed, skipping edge smoothing")
        return mask
    
    # Convert to float
    mask_float = mask.astype(float)
    
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
    
    # Threshold
    smoothed = blurred > threshold
    
    return smoothed.astype(bool)


def visualize_background_removal(
    original: Image.Image,
    mask: np.ndarray,
    cleaned: Image.Image,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Create side-by-side visualization of background removal.
    
    Args:
        original: Original image
        mask: Binary mask used
        cleaned: Image with background removed
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image
        
    Example:
        >>> vis = visualize_background_removal(original, mask, cleaned)
        >>> vis.show()
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Segmentation Mask\n(Coverage: {np.mean(mask):.1%})', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Cleaned
    axes[2].imshow(cleaned)
    axes[2].set_title('Background Removed', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # Return the saved image
        return Image.open(save_path)
    else:
        # Convert to PIL Image using buffer
        fig.canvas.draw()
        # Use buffer_rgba() instead of deprecated tostring_rgb()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        # Convert RGBA to RGB
        return Image.fromarray(buf[:, :, :3])
