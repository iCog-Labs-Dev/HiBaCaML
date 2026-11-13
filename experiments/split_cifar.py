"""Split-CIFAR-10 integrated reference A and its controls (Task-IL over five class pairs)."""

from __future__ import annotations

MODE = "cifar"
VALIDATION_FRACTION = 0.10  # per task, carved from its fit data; decisions read it, never test
AUGMENT = True              # random crop + horizontal flip on the fit split
EVAL_BATCH_SIZE = 500       # one evaluation shape for the 1,000 validation and 2,000 test images
SHARED_K_CONNECTIVITY = "kernel-only"  # shared K kernel units read only kernel coordinates

# 40 physical columns: 0-2 shared, 3-33 adaptive, 34-39 reserve; four nonshared active per task.
OVERRIDES = {
    "seed": 0,
    "epochs_per_task": 30,
    "task_local_heads": True,
    "num_tasks": 5,
    "column_pool__total_columns": 40,
    "column_pool__shared_count": 3,
    "column_pool__adaptive_count": 31,
    "column_pool__reserve_count": 6,
    "column_pool__topk_nonshared": 4,
    "reporting__log_gradient_clipping": True,
}
