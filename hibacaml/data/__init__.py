"""MNIST data helpers for HiBaCaML."""

from hibacaml.data.mnist import build_full_mnist_task, build_split_mnist_tasks
from hibacaml.types import MnistTask

__all__ = [
    "MnistTask",
    "build_full_mnist_task",
    "build_split_mnist_tasks",
]
