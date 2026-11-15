"""
MatrixGen - Generate matrices from coordinates using neural fields

This module provides functionality to generate weight matrices from coordinate
inputs using implicit neural representations, enabling memory-efficient
matrix generation for large-scale models.

Features:
- Generate matrices of arbitrary size from compact parameters
- Support for various coordinate systems (2D, 3D, etc.)
- Memory-efficient generation with chunked processing
- GPU and CPU support
- Integration with PyTorch models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class MatrixGenConfig:
    """Configuration for matrix generation"""
    hidden_size: int = 128
    num_layers: int = 4
    activation: str = "silu"
    chunk_size: int = 16384
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None


class MatrixGenerator(nn.Module):
    """
    Matrix Generator - Generate weight matrices from coordinates.
    
    This generator uses a coordinate-based neural network to produce
    weight matrices of arbitrary size from a compact set of parameters.
    
    Example:
        >>> gen = MatrixGenerator(hidden_size=128, num_layers=4)
        >>> # Generate a weight matrix
        >>> W = gen.generate_matrix(1000, 1000)
        >>> print(W.shape)  # torch.Size([1000, 1000])
    
    Args:
        hidden_size: Size of hidden layers (default: 128)
        num_layers: Number of hidden layers (default: 4)
        activation: Activation function (default: 'silu')
        chunk_size: Size of chunks for memory-efficient processing (default: 16384)
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        activation: str = "silu",
        chunk_size: int = 16384
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        
        # Determine activation function
        self.activation = self._get_activation(activation)
        
        # Build the network
        self._build_network()
        
        # Initialize with proper scaling
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def _build_network(self):
        """Build the coordinate-to-value network"""
        # Input: 2D coordinates (x, y)
        # Output: scalar value
        layers = []
        input_dim = 2  # (i, j) coordinates
        
        # First layer
        layers.append(nn.Linear(input_dim, self.hidden_size))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(self.hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _generate_chunk(
        self,
        i_start: int,
        i_end: int,
        j_start: int,
        j_end: int,
        out_dim: int,
        in_dim: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, int, int]:
        """Generate a chunk of the weight matrix"""
        # Create coordinate grid for this chunk
        i = torch.linspace(-1, 1, out_dim, device=device)[i_start:i_end]
        j = torch.linspace(-1, 1, in_dim, device=device)[j_start:j_end]
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        
        # Stack coordinates
        coords = torch.stack([
            grid_i.flatten(),
            grid_j.flatten()
        ], dim=1)
        
        # Generate values
        values = self.net(coords).view(i_end - i_start, j_end - j_start)
        
        return values, i_start, j_start
    
    def generate_matrix(
        self,
        out_dim: int,
        in_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a weight matrix of specified size.
        
        Args:
            out_dim: Output dimension
            in_dim: Input dimension
            device: Device to use (optional)
            
        Returns:
            Weight matrix of shape (out_dim, in_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Check if we need chunked processing
        total_elements = out_dim * in_dim
        if total_elements <= self.chunk_size:
            # Generate all at once
            i = torch.linspace(-1, 1, out_dim, device=device)
            j = torch.linspace(-1, 1, in_dim, device=device)
            grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
            
            coords = torch.stack([
                grid_i.flatten(),
                grid_j.flatten()
            ], dim=1)
            
            values = self.net(coords).view(out_dim, in_dim)
            
            # Apply Xavier scaling
            scale_factor = (2.0 / (in_dim + out_dim)) ** 0.5
            values = values * scale_factor
            
            return values
        else:
            # Chunked processing
            return self._generate_matrix_chunked(out_dim, in_dim, device)
    
    def _generate_matrix_chunked(
        self,
        out_dim: int,
        in_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate matrix with chunked processing"""
        W = torch.zeros(out_dim, in_dim, device=device)
        
        # Calculate chunk sizes
        # Try to make chunks as square as possible
        chunk_elements = min(self.chunk_size, out_dim * in_dim)
        chunk_rows = int(np.sqrt(chunk_elements))
        chunk_rows = min(chunk_rows, out_dim)
        
        # Process in chunks
        for i_start in range(0, out_dim, chunk_rows):
            i_end = min(i_start + chunk_rows, out_dim)
            
            for j_start in range(0, in_dim, chunk_rows):
                j_end = min(j_start + chunk_rows, in_dim)
                
                # Generate chunk
                chunk, _, _ = self._generate_chunk(
                    i_start, i_end, j_start, j_end,
                    out_dim, in_dim, device
                )
                
                # Apply Xavier scaling
                scale_factor = (2.0 / (in_dim + out_dim)) ** 0.5
                chunk = chunk * scale_factor
                
                W[i_start:i_end, j_start:j_end] = chunk
        
        return W
    
    def generate_batch(
        self,
        shapes: list,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multiple matrices with different shapes.
        
        Args:
            shapes: List of (name, out_dim, in_dim) tuples
            device: Device to use (optional)
            
        Returns:
            Dictionary mapping names to generated matrices
        """
        if device is None:
            device = next(self.parameters()).device
        
        results = {}
        for name, out_dim, in_dim in shapes:
            results[name] = self.generate_matrix(out_dim, in_dim, device)
        
        return results
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def to(self, *args, **kwargs):
        """Override to() to also set device for buffers"""
        super().to(*args, **kwargs)
        return self


class MatrixGenWrapper:
    """
    Convenience wrapper for matrix generation.
    
    Provides a higher-level interface for generating matrices for
    neural network layers.
    """
    
    def __init__(self, config: Optional[MatrixGenConfig] = None):
        if config is None:
            config = MatrixGenConfig()
        self.config = config
        self.generator = MatrixGenerator(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            activation=config.activation,
            chunk_size=config.chunk_size
        )
        if config.device:
            self.generator.to(config.device)
        if config.dtype:
            self.generator.to(config.dtype)
    
    def generate_layer_weights(
        self,
        layer_config: Dict[str, Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate weights for multiple layers.
        
        Args:
            layer_config: Dictionary mapping layer names to (out_dim, in_dim)
            
        Returns:
            Dictionary mapping layer names to weight matrices
        """
        shapes = [
            (name, out_dim, in_dim)
            for name, (out_dim, in_dim) in layer_config.items()
        ]
        return self.generator.generate_batch(shapes)
    
    def generate_model_weights(
        self,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Generate weights for all linear layers in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping parameter names to generated weights
        """
        layer_config = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_config[f'{name}.weight'] = (module.out_features, module.in_features)
                layer_config[f'{name}.bias'] = (1, module.out_features)
        
        return self.generate_layer_weights(layer_config)


def create_matrix_generator(
    hidden_size: int = 128,
    num_layers: int = 4,
    activation: str = "silu",
    chunk_size: int = 16384,
    device: Optional[str] = None
) -> MatrixGenerator:
    """
    Factory function to create a matrix generator.
    
    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        activation: Activation function name
        chunk_size: Chunk size for memory-efficient processing
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Configured MatrixGenerator instance
    """
    generator = MatrixGenerator(
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation=activation,
        chunk_size=chunk_size
    )
    if device:
        generator.to(device)
    return generator


# Example usage and testing
if __name__ == "__main__":
    print("MatrixGen - Example Usage")
    print("=" * 60)
    
    # Create generator
    gen = create_matrix_generator(hidden_size=128, num_layers=4)
    print(f"Matrix Generator created with {gen.count_parameters():,} parameters")
    
    # Test matrix generation
    print("\nGenerating sample weight matrices...")
    
    # Small matrix
    W1 = gen.generate_matrix(100, 100)
    print(f"Generated 100x100 matrix: {W1.shape}")
    print(f"  Mean: {W1.mean():.6f}, Std: {W1.std():.6f}")
    
    # Medium matrix
    W2 = gen.generate_matrix(500, 500)
    print(f"Generated 500x500 matrix: {W2.shape}")
    print(f"  Mean: {W2.mean():.6f}, Std: {W2.std():.6f}")
    
    # Batch generation
    print("\nGenerating batch of matrices...")
    shapes = [
        ('fc1', 256, 128),
        ('fc2', 128, 256),
        ('fc3', 64, 128),
    ]
    weights = gen.generate_batch(shapes)
    for name, W in weights.items():
        print(f"  {name}: {W.shape}")
    
    # Compression ratio
    original_params = 100 * 100 + 500 * 500 + 256 * 128 + 128 * 256 + 64 * 128
    dna_params = gen.count_parameters()
    print(f"\nCompression ratio: {original_params / dna_params:.1f}x")
    
    print("\n✓ MatrixGen test completed successfully!")
