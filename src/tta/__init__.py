"""
Test-Time Adaptation (TTA) methods for domain adaptation.

This package provides implementations of various TTA methods:
- TENT: Entropy minimization
- EATA: Entropy with Fisher regularization
- MEMO: Focal loss on pseudo-labels
- BN+Pseudo: Batch norm adaptation with pseudo-labeling
- MSUN: Multi-representation subdomain adaptation with uncertainty regularization
"""

from .base import TTAOptimizer
from .tent import TENTOptimizer
from .eata import EATAOptimizer
from .memo import MEMOOptimizer
from .bn_pseudo import BNPseudoOptimizer
from .msun import MSUNOptimizer

__all__ = [
    'TTAOptimizer',
    'TENTOptimizer',
    'EATAOptimizer',
    'MEMOOptimizer',
    'BNPseudoOptimizer',
    'MSUNOptimizer'
]
