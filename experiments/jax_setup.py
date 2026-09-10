"""Compatibility shim for experiments/split_mnist.py.

HiBaCaML's experiment script was written against an older FabricPC API:
    from jax_setup import set_jax_flags_before_importing_jax
    set_jax_flags_before_importing_jax(jax_platforms=...)

FabricPC >= 0.4.0 renamed and relocated this helper (see FabricPC
CHANGELOG.md, "Publish fabricpc 0.4.0 to PyPI; move JAX setup into the
package"):
    from fabricpc import setup_jax
    setup_jax(platform=...)

This shim restores the old import path/signature so the experiment
script runs unmodified against a current FabricPC install, without
patching FabricPC itself. It should live next to split_mnist.py (the
script inserts "." onto sys.path before importing it).
"""

from __future__ import annotations

from fabricpc import setup_jax


def set_jax_flags_before_importing_jax(jax_platforms: str | None = None) -> None:
    setup_jax(platform=jax_platforms)
