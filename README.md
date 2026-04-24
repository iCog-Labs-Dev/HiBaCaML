# HiBaCaML Experiments

This repository contains experiments and exploratory implementations based on the **Hierarchical Bayesian Causal Modular Learning (HiBaCaML)** framework described in *Hierarchical Bayesian Causal Modular Learning: A Two-Level Columnar Architecture for Continual Learning*.

HiBaCaML is a continual learning framework built around a **two-level modular architecture**. At the top level, a controller selects a sparse subset of component learners, or "columns", to use for a given task or context. Inside each column, a second probabilistic process separates reusable structure from task-specific information, helping the system preserve previously learned knowledge while still adapting to new tasks.

The framework is motivated by the idea that catastrophic forgetting can be reduced when learning is organized into modules with limited interference, rather than forcing all tasks to share the same parameters in a dense monolithic network. In the paper, this idea is instantiated through a columnar architecture referred to as **ColBa**, which is designed to support reuse, specialization, and controlled adaptation over time.

The experiments in this repository are implemented using [FabricPC](https://github.com/trueagi-io/FabricPC), a JAX-based predictive coding library for building modular graph-structured models. In this project, it serves as the experimental framework for exploring HiBaCaML-inspired continual learning behavior.

The work in this repository focuses on experimenting with these ideas on standard continual learning benchmarks, with particular attention to:

- `SplitMNIST`
- `SplitCIFAR`

The goal of this repo is to explore how a HiBaCaML/ColBa-style columnar, modular approach behaves on sequential task settings, and how effectively it supports continual learning without catastrophic forgetting.

## Install FabricPC

Create a Python virtual environment, then install FabricPC in editable mode:

```bash
git clone https://github.com/trueagi-io/FabricPC.git
cd FabricPC
pip install -e ".[all]"
```
