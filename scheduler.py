import math

from pydgn.training.callback.optimizer import Optimizer
from pydgn.training.callback.scheduler import EpochScheduler
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingLinearWarmup(EpochScheduler):
    def __init__(self, scheduler_class_name: str, optimizer: Optimizer, **kwargs: dict):
        assert scheduler_class_name == "torch.optim.lr_scheduler.LambdaLR"

        num_warmup_steps = kwargs["num_warmup_steps"]
        num_training_steps = kwargs["num_training_steps"]
        num_cycles = kwargs["num_cycles"]
        last_epoch = kwargs.get("last_epoch", -1)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

        self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
