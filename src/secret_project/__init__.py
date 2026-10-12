"""Attention mask guided distillation research scaffolding."""
from .core import TinyTransformer, MaskGenerator, checkpoint_indices, distillation_loss
__all__ = ["TinyTransformer", "MaskGenerator", "checkpoint_indices", "distillation_loss"]
