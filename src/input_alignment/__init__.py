"""
Input-level domain alignment methods.

These methods address domain shift by transforming target images to match
the source domain's visual statistics (color, illumination, texture).

Unlike test-time adaptation (TTA) which modifies model parameters,
input alignment preprocesses images before feeding them to the model.

Methods:
- LAB Color Space Alignment: Matches color statistics (mean/std) in LAB space
- Fourier Domain Adaptation (FDA): Swaps frequency amplitudes for style transfer
"""

from .lab_alignment import (
    compute_lab_statistics,
    apply_lab_alignment,
    LABAlignmentTransform,
    rgb_to_lab,
    visualize_lab_alignment,
    compare_lab_distributions,

)

from .fda import (
    fourier_domain_adaptation,
    apply_fda,
    visualize_fda,
    visualize_frequency_spectrum,
    compare_fda_lab,
    FDATransform
)

__all__ = [
    'compute_lab_statistics',
    'apply_lab_alignment',
    'LABAlignmentTransform',
    'rgb_to_lab',
    'visualize_lab_alignment',
    'compare_lab_distributions',
    'fourier_domain_adaptation',
    'apply_fda',
    'FDATransform',
    'visualize_fda',
    'visualize_frequency_spectrum',
    'compare_fda_lab'
]
