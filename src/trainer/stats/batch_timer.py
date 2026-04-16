import os
import torch
import src.config as config
import src.trainer.stats.base as base

trainer_stats_name="batch_timer"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return NOOPTrainerStats(conf.trainer_stats_configs.codecarbon.output_dir)

class NOOPTrainerStats(base.TrainerStats):

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.total_time = 0.0
        self.iterations = 0
        self.times = []
        self._start_event = None
        self._end_event = None

    def start_train(self) -> None: pass
    def stop_train(self) -> None: pass
    def start_step(self) -> None: pass
    def stop_step(self) -> None: pass
    def start_forward(self) -> None: pass
    def stop_forward(self) -> None: pass
    def start_backward(self) -> None: pass
    def stop_backward(self) -> None: pass
    def start_optimizer_step(self) -> None: pass
    def stop_optimizer_step(self) -> None: pass
    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None: pass
    def log_step(self) -> None: pass
    def log_loss(self, loss: torch.Tensor) -> None: pass

    def start_process_batch(self) -> None:
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event   = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop_process_batch(self) -> None:
        self._end_event.record()
        torch.cuda.synchronize()
        elapsed = self._start_event.elapsed_time(self._end_event) / 1000.0
        self.total_time += elapsed
        self.times.append(elapsed)
        self.iterations += 1

    def log_stats(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        avg = self.total_time / self.iterations
        variance = sum((t - avg) ** 2 for t in self.times) / self.iterations
        std = variance ** 0.5
        with open(os.path.join(self.output_dir, "timer.txt"), "w") as f:
            f.write(f"avg={avg}\nstd={std}\niterations={self.iterations}\n")
