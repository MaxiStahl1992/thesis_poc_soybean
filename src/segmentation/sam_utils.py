"""
SAM (Segment Anything Model) Utilities
======================================

Functions for loading and running SAM for zero-shot segmentation using Ultralytics.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import warnings

# SAM imports from Ultralytics (already installed)
try:
    from ultralytics import SAM
    from ultralytics.models.sam import SAM3SemanticPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn(
        "Ultralytics not installed. Install with: pip install -U ultralytics"
    )


def load_sam_model(
    model_path: str = 'sam3.pt',
    task: str = 'segment'
) -> SAM:
    """
    Load SAM model using Ultralytics.
    
    Args:
        model_path: Path to SAM weights (e.g., 'sam3.pt', 'sam2.pt', or local path)
        task: Task type (default: 'segment')
        
    Returns:
        Ultralytics SAM model instance
        
    Example:
        >>> model = load_sam_model('sam3.pt')
        >>> results = model.predict(source='image.jpg', points=[900, 370], labels=[1])
        
    Note:
        SAM 3 weights must be downloaded from HuggingFace:
        https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
        - vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install -U ultralytics"
        )
    
    # Load SAM model
    model = SAM(model_path)
    
    print(f"✅ Loaded SAM model: {model_path}")
    
    return model


def load_sam_semantic_predictor(
    model_path: str = 'sam3.pt',
    conf: float = 0.25,
    half: bool = True,
    save: bool = False
) -> SAM3SemanticPredictor:
    """
    Load SAM 3 Semantic Predictor for concept-based segmentation.
    
    This predictor supports text prompts and bounding box exemplars to find
    all instances of a concept in an image.
    
    Args:
        model_path: Path to SAM3 weights
        conf: Confidence threshold
        half: Use FP16 for faster inference
        save: Save results to disk
        
    Returns:
        SAM3SemanticPredictor instance
        
    Example:
        >>> predictor = load_sam_semantic_predictor('sam3.pt')
        >>> predictor.set_image('image.jpg')
        >>> results = predictor(text=['leaf', 'plant'])
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install -U ultralytics"
        )
    
    overrides = dict(
        conf=conf,
        task='segment',
        mode='predict',
        model=model_path,
        half=half,
        save=save,
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    print(f"✅ Loaded SAM3 Semantic Predictor: {model_path}")
    print(f"   Confidence: {conf}")
    print(f"   Half precision: {half}")
    
    return predictor


def segment_image_with_text(
    image_path: str,
    text_prompts: List[str],
    model_path: str = 'sam3.pt',
    conf: float = 0.25
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment an image using text prompts with SAM3.
    
    Args:
        image_path: Path to image
        text_prompts: List of text descriptions (e.g., ['leaf', 'plant'])
        model_path: Path to SAM3 weights
        conf: Confidence threshold
        
    Returns:
        Tuple of (masks, boxes) as numpy arrays, or (None, None) if no masks found
        
    Example:
        >>> masks, boxes = segment_image_with_text('leaf.jpg', ['soybean leaf'])
        >>> if masks is not None:
        ...     print(f"Found {len(masks)} leaf instances")
    """
    predictor = load_sam_semantic_predictor(model_path, conf=conf, save=False)
    predictor.set_image(image_path)
    results = predictor(text=text_prompts)
    
    if results and hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy() if hasattr(results, 'boxes') else None
        return masks, boxes
    
    return None, None


def segment_image_with_bbox(
    image_path: str,
    bboxes: List[List[float]],
    model_path: str = 'sam3.pt',
    conf: float = 0.25
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment an image using bounding box exemplars with SAM3.
    
    Args:
        image_path: Path to image
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        model_path: Path to SAM3 weights
        conf: Confidence threshold
        
    Returns:
        Tuple of (masks, boxes) as numpy arrays, or (None, None) if no masks found
        
    Example:
        >>> masks, boxes = segment_image_with_bbox('leaf.jpg', [[100, 100, 300, 300]])
        >>> if masks is not None:
        ...     print(f"Found {len(masks)} similar instances")
    """
    predictor = load_sam_semantic_predictor(model_path, conf=conf, save=False)
    predictor.set_image(image_path)
    results = predictor(bboxes=bboxes)
    
    if results and hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy() if hasattr(results, 'boxes') else None
        return masks, boxes
    
    return None, None


def segment_image(
    image: Union[str, Image.Image],
    model: SAM,
    points: Optional[List[List[int]]] = None,
    labels: Optional[List[int]] = None,
    bboxes: Optional[List[List[float]]] = None
) -> List[Dict]:
    """
    Run SAM on an image with visual prompts (SAM2-style).
    
    Args:
        image: PIL Image or path to image
        model: Ultralytics SAM model
        points: List of point coordinates [[x, y], ...]
        labels: List of labels for points (1=foreground, 0=background)
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        
    Returns:
        List of result objects from SAM prediction
        
    Example:
        >>> model = load_sam_model('sam3.pt')
        >>> results = segment_image('leaf.jpg', model, points=[[400, 300]], labels=[1])
        >>> results[0].show()
    """
    # Prepare kwargs
    kwargs = {}
    if points is not None:
        kwargs['points'] = points
    if labels is not None:
        kwargs['labels'] = labels
    if bboxes is not None:
        kwargs['bboxes'] = bboxes
    
    # Run prediction
    if isinstance(image, str):
        results = model.predict(source=image, **kwargs)
    else:
        # Convert PIL to path or save temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image.save(f.name)
            temp_path = f.name
        
        try:
            results = model.predict(source=temp_path, **kwargs)
        finally:
            os.unlink(temp_path)
    
    return results


def get_largest_mask(results) -> Optional[np.ndarray]:
    """
    Extract the largest mask from SAM results.
    
    Heuristic: For centered leaf photos, the largest mask likely corresponds to the leaf.
    
    Args:
        results: Results from SAM prediction (single result or list)
        
    Returns:
        Binary mask (H, W) as numpy array, or None if no masks
        
    Example:
        >>> model = load_sam_model('sam3.pt')
        >>> results = segment_image('leaf.jpg', model, points=[[400, 300]], labels=[1])
        >>> mask = get_largest_mask(results)
    """
    # Handle list of results
    if isinstance(results, list):
        if not results:
            return None
        result = results[0]
    else:
        result = results
    
    # Extract masks
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        if len(masks) == 0:
            return None
        
        # Find largest mask by area
        areas = [mask.sum() for mask in masks]
        largest_idx = np.argmax(areas)
        return masks[largest_idx].astype(np.uint8)
    
    return None


def get_all_masks(results) -> Optional[np.ndarray]:
    """
    Extract all masks from SAM results.
    
    Args:
        results: Results from SAM prediction
        
    Returns:
        Array of masks (N, H, W) or None if no masks
    """
    if isinstance(results, list):
        if not results:
            return None
        result = results[0]
    else:
        result = results
    
    if hasattr(result, 'masks') and result.masks is not None:
        return result.masks.data.cpu().numpy()
    
    return None


def visualize_masks(
    image: Union[str, Image.Image],
    results,
    save_path: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Visualize segmentation masks using Ultralytics' built-in plotting.
    
    Args:
        image: Original PIL Image or path to image
        results: Results from SAM prediction
        save_path: Optional path to save visualization
        
    Returns:
        PIL Image with masks overlaid, or None if no masks
        
    Example:
        >>> model = load_sam_model('sam3.pt')
        >>> results = segment_image('leaf.jpg', model, points=[[400, 300]], labels=[1])
        >>> vis = visualize_masks('leaf.jpg', results)
        >>> vis.show()
    """
    if isinstance(results, list):
        if not results:
            return None
        result = results[0]
    else:
        result = results
    
    # Use built-in visualization
    if hasattr(result, 'plot'):
        vis_array = result.plot()
        vis_image = Image.fromarray(vis_array)
        
        if save_path:
            vis_image.save(save_path)
        
        return vis_image
    
    return None


def get_mask_statistics(results) -> Dict:
    """
    Compute statistics about generated masks.
    
    Args:
        results: Results from SAM prediction
        
    Returns:
        Dictionary with statistics
        
    Example:
        >>> stats = get_mask_statistics(results)
        >>> print(f"Total masks: {stats['num_masks']}")
    """
    if isinstance(results, list):
        if not results:
            return {'num_masks': 0}
        result = results[0]
    else:
        result = results
    
    if not hasattr(result, 'masks') or result.masks is None:
        return {'num_masks': 0}
    
    masks = result.masks.data.cpu().numpy()
    areas = [mask.sum() for mask in masks]
    
    stats = {
        'num_masks': len(masks),
        'total_area': sum(areas),
        'mean_area': np.mean(areas) if areas else 0,
        'largest_area': max(areas) if areas else 0,
        'smallest_area': min(areas) if areas else 0,
    }
    
    # Add confidence scores if available
    if hasattr(result, 'boxes') and result.boxes is not None:
        if hasattr(result.boxes, 'conf'):
            confs = result.boxes.conf.cpu().numpy()
            stats['mean_confidence'] = np.mean(confs)
            stats['min_confidence'] = np.min(confs)
            stats['max_confidence'] = np.max(confs)
    
    return stats
