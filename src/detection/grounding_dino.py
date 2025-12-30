"""
GroundingDINO Symptom Detector
==============================

Language-guided detection of disease symptoms on soybean leaves.

Instead of training a detector on fixed classes, we use GroundingDINO's
zero-shot capability with natural language prompts describing visual symptoms.

Example:
    >>> detector = GroundingDINODetector()
    >>> boxes, scores = detector.detect(image, "red rust pustules")
    >>> print(f"Found {len(boxes)} rust pustules")
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class GroundingDINODetector:
    """
    Zero-shot object detector using GroundingDINO for disease symptom detection.
    
    This detector uses natural language prompts to find specific visual patterns
    (disease symptoms) without requiring bounding box annotations for training.
    
    Attributes:
        model: GroundingDINO model
        device: Compute device ('cuda', 'mps', or 'cpu')
        box_threshold: Minimum confidence for detection
        text_threshold: Minimum text similarity score
    """
    
    def __init__(
        self,
        model_config: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = 'cuda'
    ):
        """
        Initialize GroundingDINO detector.
        
        Args:
            model_config: Path to model config (if None, uses default)
            checkpoint_path: Path to checkpoint (if None, downloads)
            box_threshold: Minimum confidence for bounding boxes (0-1)
            text_threshold: Minimum text-image similarity (0-1)
            device: Device to run on ('cuda', 'mps', 'cpu')
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Import GroundingDINO (lazy import to avoid dependency issues)
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            import urllib.request
            from pathlib import Path
            
            self.load_model = load_model
            self.load_image = load_image
            self.predict = predict
        except ImportError:
            raise ImportError(
                "GroundingDINO not found. Install with:\n"
                "pip install groundingdino-py"
            )
        
        # Setup cache directory
        cache_dir = Path.home() / '.cache' / 'groundingdino'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download config if needed
        if model_config is None:
            config_path = cache_dir / 'GroundingDINO_SwinT_OGC.py'
            if not config_path.exists():
                print("Downloading GroundingDINO config...")
                config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                urllib.request.urlretrieve(config_url, config_path)
                print("✓ Config downloaded")
            model_config = str(config_path)
        
        # Download checkpoint if needed
        if checkpoint_path is None:
            checkpoint_path = cache_dir / 'groundingdino_swint_ogc.pth'
            if not checkpoint_path.exists():
                print("Downloading GroundingDINO checkpoint (1.2GB, this may take a while)...")
                checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
                print("✓ Checkpoint downloaded")
            checkpoint_path = str(checkpoint_path)
        
        print(f"Loading GroundingDINO on {device}...")
        self.model = self.load_model(model_config, checkpoint_path, device=device)
        print("✓ GroundingDINO loaded")
    
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        return_labels: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
        """
        Detect objects matching the text prompt in the image.
        
        Args:
            image: PIL Image
            text_prompt: Natural language description (e.g., "red rust pustules")
            return_labels: If True, also return detected labels
            
        Returns:
            boxes: Bounding boxes in [x1, y1, x2, y2] format (normalized 0-1)
            scores: Confidence scores for each box
            labels: Detected labels (if return_labels=True)
            
        Example:
            >>> boxes, scores = detector.detect(image, "circular lesions")
            >>> print(f"Found {len(boxes)} lesions with avg confidence {scores.mean():.2f}")
        """
        # Transform image (handle PIL Image directly)
        from groundingdino.util.utils import get_phrases_from_posmap
        import torchvision.transforms as T
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy
        image_np = np.asarray(image)
        
        # Apply transformations manually
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Resize to max dimension 800
        h, w = image_np.shape[:2]
        max_size = 1333
        size = 800
        
        scale = min(size / min(h, w), max_size / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        
        from PIL import Image as PILImage
        image_resized = PILImage.fromarray(image_np).resize((new_w, new_h), PILImage.BILINEAR)
        image_tensor = transform(image_resized)
        
        # Run detection
        boxes, logits, phrases = self.predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        # Convert to numpy
        boxes_np = boxes.cpu().numpy()  # Shape: (N, 4) in [x1, y1, x2, y2]
        scores_np = logits.cpu().numpy()  # Shape: (N,)
        
        if return_labels:
            return boxes_np, scores_np, phrases
        return boxes_np, scores_np, None
    
    def detect_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        return_labels: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray, Optional[List[str]]]]:
        """
        Detect objects in multiple images with different prompts.
        
        Args:
            images: List of PIL Images
            text_prompts: List of text prompts (one per image)
            return_labels: If True, return labels for each detection
            
        Returns:
            List of (boxes, scores, labels) tuples for each image
        """
        results = []
        for image, prompt in zip(images, text_prompts):
            result = self.detect(image, prompt, return_labels=return_labels)
            results.append(result)
        return results


def detect_symptoms(
    image: Image.Image,
    disease_prompts: Dict[str, str],
    detector: Optional[GroundingDINODetector] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = 'cuda'
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Detect symptoms for multiple diseases in a single image.
    
    This is a convenience function that runs detection for all disease types
    and returns a dictionary of results.
    
    Args:
        image: PIL Image of soybean leaf
        disease_prompts: Dict mapping disease names to symptom descriptions
            Example: {
                'Rust': 'red rust pustules on leaf',
                'Frogeye': 'circular lesions with dark rings'
            }
        detector: Existing detector (if None, creates new one)
        box_threshold: Minimum confidence for boxes
        text_threshold: Minimum text similarity
        device: Compute device
        
    Returns:
        Dictionary mapping disease names to (boxes, scores) tuples
        
    Example:
        >>> prompts = {
        ...     'Rust': 'red rust pustules',
        ...     'Frogeye': 'circular lesions with dark borders'
        ... }
        >>> results = detect_symptoms(image, prompts)
        >>> for disease, (boxes, scores) in results.items():
        ...     print(f"{disease}: {len(boxes)} symptoms detected")
    """
    # Create detector if needed
    if detector is None:
        detector = GroundingDINODetector(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
    
    # Detect for each disease
    results = {}
    for disease_name, prompt in disease_prompts.items():
        boxes, scores, _ = detector.detect(image, prompt, return_labels=False)
        results[disease_name] = (boxes, scores)
    
    return results


def visualize_detections(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3
) -> Image.Image:
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image
        boxes: Bounding boxes in [x1, y1, x2, y2] format (normalized 0-1)
        scores: Confidence scores
        labels: Optional labels for each box
        color: RGB color for boxes
        thickness: Line thickness in pixels
        
    Returns:
        Image with drawn boxes
    """
    from PIL import ImageDraw, ImageFont
    
    # Create drawing context
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Get image dimensions
    width, height = image.size
    
    # Draw each box
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # Convert normalized coords to pixels
        x1, y1, x2, y2 = box
        x1, y1 = int(x1 * width), int(y1 * height)
        x2, y2 = int(x2 * width), int(y2 * height)
        
        # Clamp coordinates to valid range and ensure x2 > x1, y2 > y1
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label if provided
        label_text = f"{labels[i]}: {score:.2f}" if labels else f"{score:.2f}"
        
        # Draw text background
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)
    
    return img_draw


def compute_symptom_severity(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_size: Tuple[int, int],
    method: str = 'weighted_count'
) -> float:
    """
    Compute disease severity score from detections.
    
    Args:
        boxes: Detected bounding boxes (normalized)
        scores: Confidence scores
        image_size: (width, height) of image
        method: Scoring method:
            - 'count': Simple count of detections
            - 'weighted_count': Count weighted by confidence
            - 'area': Total area covered by symptoms
            - 'density': Symptoms per unit leaf area
            
    Returns:
        Severity score (higher = more severe)
    """
    if len(boxes) == 0:
        return 0.0
    
    if method == 'count':
        return len(boxes)
    
    elif method == 'weighted_count':
        return np.sum(scores)
    
    elif method == 'area':
        # Compute total box area
        width, height = image_size
        total_area = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            box_width = (x2 - x1) * width
            box_height = (y2 - y1) * height
            total_area += box_width * box_height
        return total_area
    
    elif method == 'density':
        # Weighted count normalized by image area
        weighted_count = np.sum(scores)
        image_area = image_size[0] * image_size[1]
        return (weighted_count / image_area) * 100000  # Scale for readability
    
    else:
        raise ValueError(f"Unknown method: {method}")
