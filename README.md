# MatrixGen

[![CI](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**MatrixGen** is a coordinate-based matrix generation framework that creates weight matrices of arbitrary size from compact neural network parameters. Using implicit neural representations, MatrixGen enables memory-efficient generation of large-scale model weights.

### Key Features

- ✅ **Arbitrary Size**: Generate matrices of any dimension
- ✅ **Memory Efficient**: Chunked processing for large matrices
- ✅ **Compact Parameters**: Small neural network generates huge matrices
- ✅ **Flexible Architecture**: Configurable layers, activations, and sizes
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/MatrixGen.git
cd MatrixGen
pip install -e .
```

## Usage

### Basic Matrix Generation

```python
from matrix_gen import MatrixGenerator, create_matrix_generator

# Create generator
gen = create_matrix_generator(hidden_size=128, num_layers=4)

# Generate a weight matrix
W = gen.generate_matrix(1000, 1000)
print(f"Generated matrix: {W.shape}")
```

### Batch Generation

```python
# Generate multiple matrices at once
shapes = [
    ('fc1', 256, 128),
    ('fc2', 128, 256),
    ('fc3', 64, 128),
]
weights = gen.generate_batch(shapes)

for name, W in weights.items():
    print(f"{name}: {W.shape}")
```

### Model Weight Generation

```python
import torch.nn as nn
from matrix_gen import MatrixGenWrapper

# Create a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

# Generate weights for all linear layers
wrapper = MatrixGenWrapper()
weights = wrapper.generate_model_weights(MyModel())

# Load into model
model = MyModel()
model.load_state_dict(weights)
```

### Using Wrapper

```python
from matrix_gen import MatrixGenWrapper, MatrixGenConfig

# Create wrapper with custom configuration
config = MatrixGenConfig(
    hidden_size=256,
    num_layers=6,
    activation='gelu',
    chunk_size=32768
)
wrapper = MatrixGenWrapper(config)

# Generate layer weights
layer_config = {
    'weight1': (512, 256),
    'weight2': (256, 512),
    'bias1': (1, 512),
}
weights = wrapper.generate_layer_weights(layer_config)
```

## Architecture

### Coordinate-Based Generation

MatrixGen uses a neural network to map 2D coordinates to scalar values:

```
Coordinate (i, j) → Neural Network → Scalar Value
```

The network processes each coordinate independently, allowing generation of matrices of arbitrary size.

### Network Structure

The default network consists of:
- Input layer: 2 dimensions (i, j coordinates)
- Hidden layers: Configurable number with SiLU activation
- Output layer: 1 dimension (scalar value)

### Memory Efficiency

For large matrices, MatrixGen uses chunked processing:
- Configurable chunk size (default: 16384 elements)
- Automatic chunking for matrices larger than chunk size
- Seamless reconstruction of full matrix

### Xavier Scaling

All generated weights are automatically scaled using Xavier initialization to maintain proper variance.

## Compression Ratios

| Matrix Size | Generator Parameters | Compression Ratio |
|-------------|---------------------|-------------------|
| 100x100     | ~10K                | ~1000x            |
| 1000x1000   | ~10K                | ~100,000x         |
| 10000x10000 | ~10K                | ~10,000,000x      |

*The generator uses only ~10K parameters regardless of the output matrix size.*

## API Reference

### MatrixGenerator

```python
MatrixGenerator(
    hidden_size: int = 128,
    num_layers: int = 4,
    activation: str = "silu",
    chunk_size: int = 16384
)
```

**Methods:**
- `generate_matrix(out_dim: int, in_dim: int) -> Tensor`: Generate a weight matrix
- `generate_batch(shapes: list) -> Dict`: Generate multiple matrices
- `count_parameters() -> int`: Count generator parameters

**Supported Activations:**
- `silu` (default)
- `relu`
- `gelu`
- `tanh`
- `sigmoid`
- `elu`
- `leaky_relu`

### MatrixGenWrapper

High-level wrapper for easier use:

```python
wrapper = MatrixGenWrapper(config=MatrixGenConfig())
```

**Methods:**
- `generate_layer_weights(layer_config: Dict) -> Dict`: Generate weights for layers
- `generate_model_weights(model: nn.Module) -> Dict`: Generate weights for model

### create_matrix_generator()

Factory function for quick generator creation:

```python
gen = create_matrix_generator(
    hidden_size=128,
    num_layers=4,
    activation='silu',
    device='cuda'
)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Implicit neural representations
- Coordinate-based neural networks
- Neural fields
- Memory-efficient deep learning
