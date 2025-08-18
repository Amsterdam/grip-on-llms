"""
Social bias benchmarks module.

This module contains benchmarks for evaluating social biases in language models,
with a focus on Dutch cultural context and municipal governance considerations.
"""

from .base import SocialBiasBenchmark
from .bzk_social_bias import BZKSocialBias

__all__ = ["SocialBiasBenchmark", "BZKSocialBias"]