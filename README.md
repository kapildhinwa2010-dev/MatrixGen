# MatrixGen

MatrixGen is a small Python/PyTorch prototype for producing two-dimensional tensors from row and column coordinates. A multilayer perceptron maps normalized coordinate pairs to scalar values; its parameters are randomly initialized, so generated matrices are not trained or reconstructed weights.

## Status

The repository contains a working implementation and unit tests. Its package metadata labels the project **Beta**. It should be treated as an experimental matrix-generation prototype, not as a validated compression method or a production-ready way to create model weights. No task-quality, reconstruction-quality, or performance benchmarks are provided here.

## What is implemented

- `MatrixGenerator` creates a matrix of a requested `(out_dim, in_dim)` shape by evaluating a coordinate-to-scalar PyTorch network.
- Configurable hidden width, depth, chunk size, and activation (`silu`, `relu`, `gelu`, `tanh`, `sigmoid`, `elu`, or `leaky_relu`).
- Chunked coordinate evaluation for larger outputs, as well as batch generation for named matrix shapes.
- `MatrixGenConfig` and `MatrixGenWrapper` helpers for requesting matrices shaped for linear layers; `generate_model_weights` returns generated tensors for a model's linear layers.
- A factory function, `create_matrix_generator`, and a runnable example in `matrix_gen.py`.

## Requirements and use

Python 3.10 or later is declared in the project metadata. From the repository root, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the module's small demonstration:

```bash
python matrix_gen.py
```

Or import the module from the repository checkout:

```python
from matrix_gen import MatrixGenerator

generator = MatrixGenerator(hidden_size=64, num_layers=3)
weights = generator.generate_matrix(out_dim=128, in_dim=64)
print(weights.shape)  # torch.Size([128, 64])

batch = generator.generate_batch([
    ("projection", 128, 64),
    ("output", 10, 128),
])
```

The example generates values from the network's random initialization. It does not train the generator or make the resulting tensors useful for a particular model. The repository has no configured command-line entry point; `python matrix_gen.py` runs only the included demonstration.

## Tests

The test suite can be run from the repository root with:

```bash
python -m pytest -q
```

On commit `14116ff14615cbe462811f550f0227482895a261`, this command was run locally with Python 3.12.3 and completed with **26 passed**. The GitHub Actions workflow is configured to run pytest on Python 3.10, 3.11, and 3.12; no remote workflow result is asserted here.

## Limitations

- There is no training, fitting, checkpoint-loading, or reconstruction workflow in the implementation. Generated tensors are outputs of the current randomly initialized network, not learned replacements for existing weights.
- The network accepts only two-dimensional row/column coordinates and produces one scalar per coordinate pair; broader coordinate systems are not implemented.
- `generate_model_weights` returns a dictionary of tensors rather than assigning them to the input model. Its generated bias entries have shape `(1, out_features)`, which differs from the usual one-dimensional `nn.Linear.bias` shape.
- Chunking limits intermediate coordinate evaluation, but the complete output matrix is still allocated in memory.
- Tests exercise generation and wrapper behavior, not model accuracy, compression quality, large-scale speed, or CUDA execution. No benchmark or controlled quality results are reported.

## License

The repository's `LICENSE` file contains the MIT License, copyright 2026 Kapil Dhinwa. See [LICENSE](LICENSE) for the terms.
