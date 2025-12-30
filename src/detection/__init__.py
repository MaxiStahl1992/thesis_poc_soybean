"""
Detection Module
================

Language-guided symptom detection using Vision-Language models.
"""

from .grounding_dino import GroundingDINODetector, detect_symptoms

__all__ = ['GroundingDINODetector', 'detect_symptoms']
