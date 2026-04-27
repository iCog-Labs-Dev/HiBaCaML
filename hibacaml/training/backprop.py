"""Backprop runner for the HiBaCaML subsystem."""

from __future__ import annotations

import copy
from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from hibacaml.training.trainer import (
    HiBaCaMLTrainer,
    _loss_dict_from_vector,
    _loss_vector_from_state,
)


class HiBaCaMLBackpropRunner(HiBaCaMLTrainer):
    """Backprop runner that differentiates through predictive-coding inference."""

    def clone(self) -> "HiBaCaMLBackpropRunner":
        cloned = HiBaCaMLBackpropRunner(
            cfg=copy.deepcopy(self.cfg),
            structure=self.structure,
            params=copy.deepcopy(self.params),
            tasks=list(self.tasks.values()),
            optimizer=self.optimizer,
            rng_key=self.rng_key,
            persistent_state=copy.deepcopy(self.persistent_state),
            create_run_logger=False,
        )
        cloned.opt_state = copy.deepcopy(self.opt_state)
        cloned.persistent_state.params = cloned.params
        cloned.persistent_state.opt_state = cloned.opt_state
        cloned.current_phi = copy.deepcopy(self.current_phi)
        cloned.current_nonshared = tuple(self.current_nonshared)
        cloned._jit_run_batch_inference = self._jit_run_batch_inference
        cloned._jit_compute_pc_gradients = self._jit_compute_pc_gradients
        cloned._jit_eval_batch = self._jit_eval_batch
        cloned._eval_cache = copy.deepcopy(self._eval_cache)
        return cloned

    def compute_pc_gradients(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params=None,
    ):
        params = params if params is not None else self.params
        clamps = self._build_clamps(batch, task, nonshared, params=params)
        self.rng_key, state_key = jax.random.split(self.rng_key)

        def loss_fn(p):
            final_state = self._jit_run_batch_inference(p, clamps, state_key)
            losses = _loss_vector_from_state(final_state, self.structure)
            return losses[-1], (final_state, losses)

        (total_loss, (final_state, loss_vector)), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params)
        self._last_graph_state = final_state
        losses = _loss_dict_from_vector(loss_vector)
        losses["total"] = float(total_loss)
        return grads, losses, final_state
