# MatrixGen

[![CI](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/MatrixGen/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Coordinate-based generator that emits weight matrices of arbitrary size.

## What it does

MatrixGenerator maps 2D coordinates through a small MLP (SiLU by
default) to scalar values, producing matrices of any requested shape
from roughly 10K generator parameters, with chunked processing for
large shapes.

## Installation

    pip install torch numpy pytest

## Usage

    from matrix_gen import create_matrix_generator

    gen = create_matrix_generator(hidden_size=128, num_layers=4)
    W = gen.generate_matrix(1000, 1000)

    shapes = [("fc1", 256, 128), ("fc2", 128, 64)]
    weights = gen.generate_batch(shapes)

## API

- MatrixGenerator(hidden_size=128, num_layers=4, activation="silu",
  chunk_size=16384): generate_matrix, generate_batch, count_parameters
- Activations: silu, relu, gelu, tanh, sigmoid, elu, leaky_relu
- MatrixGenWrapper / MatrixGenConfig: generate_layer_weights,
  generate_model_weights
- create_matrix_generator(...) factory

## Honest results and limitations

- The table often quoted (up to 10,000,000x) is parameter arithmetic:
  output elements divided by ~10K generator params. It is not a
  compression result and implies nothing about quality.
- An untrained generator outputs a smooth random field, not useful
  weights. Making the matrices useful requires fitting the generator
  per matrix, which costs at least as much as the fitting itself saves
  for small and medium matrices.
- No task-accuracy or reconstruction-error numbers are reported because
  none were measured in a controlled experiment.
- Generation of very large matrices is slow on CPU (minutes for
  10000x10000-scale shapes).

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that coordinate-generated weights
  did not reconstruct trained weights acceptably without per-layer
  training, and the related carrier-mode recall failed about half of
  noise trials. The robust design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
