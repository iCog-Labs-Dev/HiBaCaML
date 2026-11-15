# HiBaCaML Experiments

Continual-learning experiments based on [*Hierarchical Bayesian Causal Modular
Learning: A Two-Level Columnar Architecture for Continual Learning*](https://link.springer.com/chapter/10.1007/978-3-032-33010-9_21).
The paper introduces **HiBaCaML**, in which a controller activates a sparse set of
columns for each task, and each column
separates reusable structure from task-specific information, so new tasks can be
learned with limited forgetting of earlier ones. The models are built with
[FabricPC](https://github.com/trueagi-io/FabricPC), a JAX predictive-coding library.

## Setup

```bash
pip install -r requirements.txt
```

## Experiments

Run from the repository root; results are written under `runs/`.

| Experiment | Command |
|---|---|
| Split-MNIST | `python -m experiments.split_mnist` |
| Full-MNIST | `python -m experiments.mnist` |
| Full-CIFAR-10 (GPU) | `python -m experiments.cifar --seed 0` |
| Split-CIFAR-10 (GPU) | `python -m experiments.split_cifar --seed 0` |

The method, code structure and experiment details are documented in
[docs/Implementation_details.md](docs/Implementation_details.md).
