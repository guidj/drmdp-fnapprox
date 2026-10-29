"""Adapters for estimator tests whose feature functions are FTOp test doubles.

Production adapters perform the encoding so tests cannot silently use a
second discrete-index or vector conversion. Successor-enabled test doubles
exercise estimator accumulation, not production successor compilation.
"""

import numpy as np

from drmdp import rewardfeat, transform


class FTOpHook:
    """Delegates feature encoding to the production FTOp adapter."""

    def __init__(self, ft_op: transform.FTOp, successor: bool = False):
        current = rewardfeat.FTOpFeatures(ft_op)
        self.features: rewardfeat.TransitionFeatureFn = current
        if successor:
            self.features = rewardfeat.CurrentAndSuccessorFeatures(
                current, rewardfeat.FTOpFeatures(ft_op, successor=True)
            )

    @property
    def feature_dim(self) -> int:
        return self.features.feature_dim

    def __call__(self, transition: rewardfeat.Transition) -> np.ndarray:
        return self.features(transition)
