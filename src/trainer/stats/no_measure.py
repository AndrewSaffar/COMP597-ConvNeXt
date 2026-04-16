import os
import time

import src.config as config
import src.trainer.stats.base as base
import torch

trainer_stats_name="no_measure"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return NOOPTrainerStats(conf.trainer_stats_configs.codecarbon.output_dir)

class NOOPTrainerStats(base.TrainerStats):
    """NOOP Trainer stats to ignore data accumulation.

    This class implements the `TrainerStats` interface. All the methods are 
    NOOP so that training can be done with accumulating statistics.

    """

    def __init__(self, output_dir : str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.end_time = None
        self.start_time = None

    def start_train(self) -> None:
        self.start_time = time.perf_counter()

    def stop_train(self) -> None:
        self.end_time = time.perf_counter()

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass
    
    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.end_time is None:
            self.end_time = time.perf_counter()
        with open(os.path.join(self.output_dir, "timer.txt"), "w") as f:
            f.write(str(self.end_time - self.start_time))

    def log_loss(self, loss: torch.Tensor) -> None:
        pass
