"""
Batch Processing for Datasets
=============================

Functions for processing entire datasets with SAM segmentation.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from tqdm.auto import tqdm
from PIL import Image
import json
import numpy as np


def process_dataset(
    source_dir: str,
    output_dir: str,
    model,
    text_prompt: Optional[str] = None,
    background_color: tuple = (0, 0, 0),
    merge_threshold: float = 0.02,
    use_yolo_detection: bool = True,
    save_metadata: bool = True,
    file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
) -> Dict:
    """
    Process an entire dataset directory with SAM segmentation.
    
    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for processed images
        model: Ultralytics SAM model or path to model
        text_prompt: Optional text prompt for SAM3 (e.g., 'leaf', 'plant')
        background_color: Background color for removed pixels
        merge_threshold: Merge all masks above this fraction of image area (default: 2%)
        use_yolo_detection: Use multi-point grid strategy (default: True, recommended)
        save_metadata: Save processing metadata to JSON
        file_extensions: Valid image file extensions
        
    Returns:
        Dictionary with processing statistics
        
    Example:
        >>> from src.segmentation import load_sam_model
        >>> model = load_sam_model('sam_b.pt')
        >>> # Multi-point grid (recommended for complete coverage)
        >>> stats = process_dataset(
        ...     source_dir='data/MH',
        ...     output_dir='data/MH_Segmented',
        ...     model=model,
        ...     use_yolo_detection=True,  # Enables multi-point grid
        ...     merge_threshold=0.02
        ... )
    """
    from .background_removal import process_image_with_sam
    from .sam_utils import load_sam_model
    
    # Load model if path provided
    if isinstance(model, str):
        model = load_sam_model(model)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in file_extensions:
        image_files.extend(source_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Statistics
    stats = {
        'total_images': len(image_files),
        'processed': 0,
        'failed': 0,
        'total_masks_found': 0,
        'avg_mask_coverage': 0.0,
        'failed_files': []
    }
    
    metadata_list = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get relative path to preserve directory structure
            rel_path = img_path.relative_to(source_path)
            out_path = output_path / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process with SAM
            cleaned_image, mask, metadata = process_image_with_sam(
                str(img_path),
                model,
                background_color=background_color,
                text_prompt=text_prompt,
                merge_threshold=merge_threshold,
                use_yolo_detection=use_yolo_detection
            )
            
            # Save cleaned image
            cleaned_image.save(out_path, quality=95)
            
            # Update statistics
            stats['processed'] += 1
            stats['total_masks_found'] += metadata['num_masks_found']
            stats['avg_mask_coverage'] += metadata['mask_coverage']
            
            # Store metadata
            if save_metadata:
                metadata_list.append({
                    'file': str(rel_path),
                    'source': str(img_path),
                    'output': str(out_path),
                    **metadata
                })
            
        except Exception as e:
            print(f"\n❌ Failed to process {img_path}: {e}")
            stats['failed'] += 1
            stats['failed_files'].append(str(img_path))
    
    # Compute averages
    if stats['processed'] > 0:
        stats['avg_mask_coverage'] /= stats['processed']
        stats['avg_masks_per_image'] = stats['total_masks_found'] / stats['processed']
    
    # Save metadata
    if save_metadata and metadata_list:
        metadata_path = output_path / 'segmentation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'per_image_metadata': metadata_list
            }, f, indent=2)
        print(f"\n💾 Saved metadata to: {metadata_path}")
    
    return stats


def create_segmented_dataset(
    dataset_name: str,
    data_root: str,
    output_root: str,
    model,
    text_prompt: Optional[str] = None,
    background_color: tuple = (0, 0, 0),
    merge_threshold: float = 0.02,
    use_yolo_detection: bool = True,
    class_folders: Optional[List[str]] = None
) -> Dict:
    """
    Create a segmented version of a classification dataset.
    
    Preserves the directory structure: output_root/class_name/image.jpg
    
    Args:
        dataset_name: Name of dataset ('MH', 'ASDID', etc.)
        data_root: Root directory of source dataset
        output_root: Root directory for output dataset
        model: Ultralytics SAM model or path to model
        text_prompt: Optional text prompt for SAM3 (e.g., 'leaf', 'plant')
        background_color: Background color
        merge_threshold: Merge all masks above this fraction of image area (default: 2%)
        use_yolo_detection: Use multi-point grid strategy (default: True, recommended)
        class_folders: List of class folder names (if None, process all subdirs)
        
    Returns:
        Processing statistics
        
    Example:
        >>> from src.segmentation import load_sam_model
        >>> model = load_sam_model('sam_b.pt')
        >>> # Multi-point grid for complete coverage
        >>> stats = create_segmented_dataset(
        ...     dataset_name='MH',
        ...     data_root='data/MH-SoyaHealthVision/Soyabean_Leaf_Image_Dataset',
        ...     output_root='data/MH_Segmented',
        ...     model=model,
        ...     use_yolo_detection=True,  # Enables multi-point grid (recommended)
        ...     merge_threshold=0.02,
        ...     class_folders=['Healthy_Soyabean', 'Soyabean_Rust']
        ... )
    """
    data_path = Path(data_root)
    output_path = Path(output_root)
    
    # Auto-detect class folders if not specified
    if class_folders is None:
        class_folders = [d.name for d in data_path.iterdir() if d.is_dir()]
        print(f"Auto-detected class folders: {class_folders}")
    
    # Process each class folder
    all_stats = {
        'dataset_name': dataset_name,
        'total_images': 0,
        'total_processed': 0,
        'total_failed': 0,
        'per_class_stats': {}
    }
    
    for class_name in class_folders:
        print(f"\n{'='*70}")
        print(f"Processing class: {class_name}")
        print(f"{'='*70}")
        
        class_source = data_path / class_name
        class_output = output_path / class_name
        
        if not class_source.exists():
            print(f"⚠️ Warning: {class_source} does not exist, skipping...")
            continue
        
        # Process this class
        class_stats = process_dataset(
            source_dir=str(class_source),
            output_dir=str(class_output),
            model=model,
            text_prompt=text_prompt,
            background_color=background_color,
            merge_threshold=merge_threshold,
            use_yolo_detection=use_yolo_detection,
            save_metadata=True
        )
        
        # Update overall stats
        all_stats['total_images'] += class_stats['total_images']
        all_stats['total_processed'] += class_stats['processed']
        all_stats['total_failed'] += class_stats['failed']
        all_stats['per_class_stats'][class_name] = class_stats
    
    # Save overall statistics
    stats_path = output_path / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n💾 Saved overall statistics to: {stats_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SEGMENTATION COMPLETE - {dataset_name}")
    print(f"{'='*70}")
    print(f"Total images: {all_stats['total_images']}")
    print(f"Successfully processed: {all_stats['total_processed']}")
    print(f"Failed: {all_stats['total_failed']}")
    print(f"Success rate: {all_stats['total_processed']/all_stats['total_images']*100:.1f}%")
    print(f"\nOutput directory: {output_path}")
    print(f"{'='*70}")
    
    return all_stats


def compare_original_vs_segmented(
    original_dir: str,
    segmented_dir: str,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Create side-by-side comparison visualization of original vs segmented images.
    
    Args:
        original_dir: Directory with original images
        segmented_dir: Directory with segmented images
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
        
    Example:
        >>> compare_original_vs_segmented(
        ...     'data/MH/Soyabean_Rust',
        ...     'data/MH_Segmented/Soyabean_Rust',
        ...     num_samples=5
        ... )
    """
    import matplotlib.pyplot as plt
    
    orig_path = Path(original_dir)
    seg_path = Path(segmented_dir)
    
    # Find matching image files
    orig_files = list(orig_path.glob('*.[jJ][pP][gG]'))
    if not orig_files:
        orig_files = list(orig_path.glob('*.png'))
    
    # Sample random files
    import random
    sample_files = random.sample(orig_files, min(num_samples, len(orig_files)))
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, orig_file in enumerate(sample_files):
        # Load original
        orig_img = Image.open(orig_file)
        
        # Load segmented (same filename)
        seg_file = seg_path / orig_file.name
        if seg_file.exists():
            seg_img = Image.open(seg_file)
        else:
            seg_img = Image.new('RGB', orig_img.size, color=(255, 0, 0))
            print(f"⚠️ Segmented file not found: {seg_file}")
        
        # Plot
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original - {orig_file.name}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(seg_img)
        axes[i, 1].set_title(f'Segmented - {orig_file.name}', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison to: {save_path}")
    
    plt.show()


def verify_dataset_structure(output_dir: str) -> bool:
    """
    Verify that the segmented dataset has the correct structure.
    
    Args:
        output_dir: Segmented dataset directory
        
    Returns:
        True if structure is valid
        
    Example:
        >>> is_valid = verify_dataset_structure('data/MH_Segmented')
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Output directory does not exist: {output_path}")
        return False
    
    # Find class folders
    class_folders = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_folders:
        print(f"❌ No class folders found in {output_path}")
        return False
    
    print(f"✅ Found {len(class_folders)} class folders:")
    
    total_images = 0
    for class_folder in class_folders:
        images = list(class_folder.glob('*.[jJ][pP][gG]'))
        images.extend(list(class_folder.glob('*.png')))
        total_images += len(images)
        print(f"   • {class_folder.name}: {len(images)} images")
    
    print(f"\n✅ Total images: {total_images}")
    
    # Check metadata
    metadata_file = output_path / 'dataset_statistics.json'
    if metadata_file.exists():
        print(f"✅ Found metadata file: {metadata_file}")
    else:
        print(f"⚠️ Metadata file not found (optional)")
    
    return True
