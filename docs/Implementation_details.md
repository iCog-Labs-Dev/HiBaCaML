# HiBaCaML Implementation Details

The implementation documentation is split by the question it answers:

| Document | Read it for |
|---|---|
| [ALGORITHM.md](ALGORITHM.md) | What the method does and how it maps to the HiBaCaML/ColBa papers: columns and shells, certificates, the composer and hierarchy losses, the MNIST support controller, the CIFAR-10 mechanisms, deviations from the papers, and pseudocode. |
| [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | Every module and what it owns, the order of operations in one training update, how a run flows through the code, the two learners (backprop and predictive coding), and a suggested reading order. |
| [EXPERIMENTS.md](EXPERIMENTS.md) | The four runners (Split-MNIST, Full-MNIST, Full-CIFAR-10, Split-CIFAR-10): data, configuration, how to run them, what they write, and reference results. |
| [COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md) | The Split-CIFAR-10 controllers in full: exact write protection, restricted column reuse at task boundaries, and audited shell adaptation during training. |

Quick start:

```bash
pip install -r requirements.txt
python -m experiments.split_mnist                 # Split-MNIST (settings at the top of the file)
python -m experiments.mnist                       # Full-MNIST
python -m experiments.cifar --seed 0              # Full-CIFAR-10 (GPU)
python -m experiments.split_cifar --seed 0        # Split-CIFAR-10 (GPU)
```

Each run writes its records under `runs/`.
