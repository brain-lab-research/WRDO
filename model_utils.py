"""
Utilities for model and learnable weights.
"""

import torch
import torch.nn.functional as F
from typing import List
from heretic.model import Model, AbliterationParameters


class LearnableDirectionWeights(torch.nn.Module):
    """
    Learnable coefficients for weighted sum of refusal directions.

    Supports two parameterizations:
    - dense: one coefficient matrix (n_layers+1, hidden_size) per direction
    - scalar: one scalar coefficient per direction
    """
    def __init__(
        self, 
        n_directions: int, 
        n_layers: int, 
        hidden_size: int, 
        init_scale: float = 0.1,
        init_type: str = "zero",
        topic_idx: int = 0,
        mode: str = "dense",
    ):
        """
        Args:
            n_directions: Number of refusal directions
            n_layers: Number of layers (excluding embeddings)
            hidden_size: Hidden size
            init_scale: Init scale (for random init only)
            init_type: "zero", "topic", or "average"
            topic_idx: Direction index for "topic" init
            mode: "dense" or "scalar"
        """
        super().__init__()
        self.n_directions = n_directions
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.mode = mode

        if self.mode not in {"dense", "scalar"}:
            raise ValueError(
                f"Unsupported LearnableDirectionWeights mode '{self.mode}'. "
                "Expected 'dense' or 'scalar'."
            )

        if init_type == "zero":
            init_weights = self._zeros()
        elif init_type == "topic":
            init_weights = self._zeros()
            if self.mode == "scalar":
                init_weights[topic_idx] = 1.0
            else:
                init_weights[topic_idx] = torch.ones(n_layers + 1, hidden_size)
        elif init_type == "average":
            init_weights = self._ones() / n_directions
        else:
            init_weights = torch.randn(self._weight_shape()) * init_scale

        self.weights = torch.nn.Parameter(init_weights)

    def _weight_shape(self) -> tuple[int, ...]:
        if self.mode == "scalar":
            return (self.n_directions,)
        return (self.n_directions, self.n_layers + 1, self.hidden_size)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self._weight_shape())

    def _ones(self) -> torch.Tensor:
        return torch.ones(self._weight_shape())

    def _validate_weights_shape(self, weights: torch.Tensor) -> None:
        expected_dense_shape = (self.n_directions, self.n_layers + 1, self.hidden_size)
        expected_scalar_shape = (self.n_directions,)

        if weights.ndim == 1:
            if tuple(weights.shape) != expected_scalar_shape:
                raise ValueError(
                    f"Scalar direction weights must have shape {expected_scalar_shape}, "
                    f"got {tuple(weights.shape)}."
                )
            return

        if weights.ndim == 3:
            if tuple(weights.shape) != expected_dense_shape:
                raise ValueError(
                    f"Dense direction weights must have shape {expected_dense_shape}, "
                    f"got {tuple(weights.shape)}."
                )
            return

        raise ValueError(
            "Direction weights must be either a scalar vector with shape "
            f"{expected_scalar_shape} or a dense tensor with shape {expected_dense_shape}, "
            f"got ndim={weights.ndim} and shape={tuple(weights.shape)}."
        )

    def combine_with_weights(
        self,
        refusal_directions: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a weighted sum of refusal directions using an arbitrary weight tensor.

        Args:
            refusal_directions: List of tensors (n_layers+1, hidden_size)
            weights: Tensor with shape (n_directions, n_layers+1, hidden_size)

        Returns:
            Weighted, layer-wise normalized sum with shape (n_layers+1, hidden_size)
        """
        if len(refusal_directions) != self.n_directions:
            raise ValueError(
                f"Expected {self.n_directions} refusal directions, got {len(refusal_directions)}."
            )

        directions_tensor = torch.stack(refusal_directions, dim=0)
        expected_directions_shape = (self.n_directions, self.n_layers + 1, self.hidden_size)
        if tuple(directions_tensor.shape) != expected_directions_shape:
            raise ValueError(
                f"Refusal directions must stack to shape {expected_directions_shape}, "
                f"got {tuple(directions_tensor.shape)}."
            )

        self._validate_weights_shape(weights)
        weights = weights.to(device=directions_tensor.device, dtype=directions_tensor.dtype)
        if weights.ndim == 1:
            weights = weights[:, None, None]

        weighted_directions = weights * directions_tensor
        combined_direction = weighted_directions.sum(dim=0)
        combined_direction = F.normalize(combined_direction, p=2, dim=1)
        return combined_direction
    
    def forward(self, refusal_directions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of refusal directions.
        
        Args:
            refusal_directions: List of tensors (n_layers+1, hidden_size)
        
        Returns:
            Weighted sum (n_layers+1, hidden_size)
        """
        return self.combine_with_weights(refusal_directions, self.weights)


def apply_abliteration_with_hyperparams(
    model: Model,
    refusal_directions: torch.Tensor,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
    n_layers: int
) -> None:
    """
    Apply abliteration with given hyperparameters.
    
    Args:
        model: Model to modify
        refusal_directions: Refusal directions
        max_weight: Max shift weight
        max_weight_position: Position of max weight (fraction of layers, 0-1)
        min_weight: Min weight
        min_weight_distance: Distance for weight decay (fraction of layers)
        n_layers: Number of layers
    """
    max_weight_pos_abs = max_weight_position * (n_layers - 1)
    min_weight_dist_abs = min_weight_distance * (n_layers - 1)
    
    parameters = {}
    for component in model.get_abliterable_components():
        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_pos_abs,
            min_weight=min_weight,
            min_weight_distance=min_weight_dist_abs
        )
    
    with torch.no_grad():
        model.abliterate(refusal_directions, direction_index=None, parameters=parameters)
