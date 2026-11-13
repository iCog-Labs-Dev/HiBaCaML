"""Static Full-CIFAR-10 experiment (R6): joint ten-class capacity of the 4+5 active layout."""

from __future__ import annotations

MODE = "cifar"
VALIDATION_FRACTION = 0.10
AUGMENT = True              # random crop + horizontal flip on the fit split
EVAL_BATCH_SIZE = 500       # one evaluation shape for the 5,000 validation and 10,000 test images
SHARED_K_CONNECTIVITY = "dense"

# 40 physical columns: 0-3 shared, 4-8 adaptive and always active, 9-39 inactive reserve.
OVERRIDES = {
    "seed": 0,
    "epochs_per_task": 60,
    "task_local_heads": False,
    "num_tasks": 1,
    "column_pool__total_columns": 40,
    "column_pool__shared_count": 4,
    "column_pool__adaptive_count": 5,
    "column_pool__reserve_count": 31,
    "column_pool__topk_nonshared": 5,
}
