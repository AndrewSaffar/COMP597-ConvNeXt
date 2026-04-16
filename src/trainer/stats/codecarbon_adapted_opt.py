import time
from typing import List
import logging
import os
import csv
import pandas as pd
import src.config as config
import src.trainer.stats.base as base
import torch
import pynvml
from src.trainer.stats.utils import RunningStat
import psutil

logger = logging.getLogger(__name__)

trainer_stats_name="codecarbon_adapted_opt"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon trainer stats. Using default PyTorch device")
        device = torch.get_default_device() 
    return CodeCarbonStats(device, conf.trainer_stats_configs.codecarbon.run_num, conf.trainer_stats_configs.codecarbon.project_name, conf.trainer_stats_configs.codecarbon.output_dir)

class GpuUtilisationStat:

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._phases = ["step", "forward", "backward", "optimizer"]
        self.gpu_util: dict[str, RunningStat] = {p: RunningStat() for p in self._phases}
        self.mem_used: dict[str, RunningStat] = {p: RunningStat() for p in self._phases}
        self.cpu_util: dict[str, RunningStat] = {p: RunningStat() for p in self._phases}
        self._proc = psutil.Process()
        self._proc.cpu_percent(interval=None)

    def _query_gpu_util(self) -> int:
        rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return rates.gpu 

    def _query_mem_used(self) -> int:
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used
    def record(self, phase: str) -> None:
        if phase not in self._phases:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {self._phases}.")
        self.gpu_util[phase].update(self._query_gpu_util())
        self.mem_used[phase].update(self._query_mem_used())
        self.cpu_util[phase].update(self._proc.cpu_percent(interval=None) / psutil.cpu_count())


    def get_last_gpu_util(self, phase: str) -> int:
        return self.gpu_util[phase].get_last()

    def get_avg_gpu_util(self, phase: str) -> float:
        return self.gpu_util[phase].get_average()

    def get_last_mem_used(self, phase: str) -> int:
        return self.mem_used[phase].get_last()

    def get_avg_mem_used(self, phase: str) -> float:
        return self.mem_used[phase].get_average()



    def log_analysis(self, phase: str) -> None:

        """Print quantile breakdown of GPU utilisation and memory for *phase*."""

        print(f"\n=== GPU utilisation (%) — phase: {phase} ===")

        self.gpu_util[phase].log_analysis()

        print(f"\n=== GPU memory used (bytes) — phase: {phase} ===")

        self.mem_used[phase].log_analysis()

class CodeCarbonStats(base.TrainerStats):
    """Provides energy consumed and carbon emitted during model training. 
    
    This class measures the energy consumption and carbon emissions of the 
    forward pass, backward pass, and optimiser step, as well as of the training 
    as a whole.

    Implemented using the CodeCarbon library: 
    https://mlco2.github.io/codecarbon/.

    Parameters
    ----------
    device
        A PyTorch device which will be the targets of the measurements.
    run_num
        Used to number different experiments in case their measurements get 
        merged into a single file.
    project_name
        Used by CodeCarbon to identify the experiments. 

    """

    def __init__(self, device : torch.device, run_num : int, project_name : str, output_dir : str) -> None: 
        
        # Track current iteration number in the training loop
        self.iteration = 0

        self.end_time = None
        self.start_time = None

        self.STEP_WINDOW = 10
        
        # CUDA device indicates the current GPU assigned to this process (0, 1, 2, ...)
        self.device = device
        # tracking the run number to distinguish between different parameter settings
        self.run_num = run_num
        run_number = f"run_{run_num}_"
        # GPU ranks - wrap in torch.device
        gpu_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        # log the losses
        self.losses = []
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        pynvml.nvmlInit()
        self.gpu_util_tracker = GpuUtilisationStat(gpu_index=gpu_id)

    def start_train(self) -> None:
        self.start_time = time.perf_counter()
        torch.cuda.synchronize(self.device)

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.end_time = time.perf_counter()

    def start_step(self) -> None:
        pass
        
    def stop_step(self) -> None:
        self.iteration += 1

    def start_forward(self) -> None: 
        pass

    def stop_forward(self) -> None: 
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        if self.iteration % self.STEP_WINDOW == 1:
            torch.cuda.synchronize(self.device)
            self.gpu_util_tracker.record("optimizer")

    def start_save_checkpoint(self) -> None:
        logger.warning(f"Method 'start_save_checkpoint' is not implemented for '{self.__class__.__name__}'.")

    def stop_save_checkpoint(self) -> None:
        logger.warning(f"Method 'stop_save_checkpoint' is not implemented for '{self.__class__.__name__}'.")

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        """
        Log the loss statistics to an external file.
        """
        # losses as dataframe
        df = pd.DataFrame([[x["task_name"], x["loss"].item()] for x in self.losses])
        
        # save to file ({output_dir}/losses/run_{run_num}_cc_loss_rank_{gpu_id}.csv)
        run_number = f"run_{self.run_num}_"
        gpu_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(self.device)
        losses_dir = os.path.join(self.output_dir, "losses")
        os.makedirs(losses_dir, exist_ok=True)
        save_file_path = os.path.join(losses_dir, f"{run_number}cc_loss_rank_{gpu_id}.csv")
        df.to_csv(save_file_path, index=False)

        if self.end_time is None:
            self.end_time = time.perf_counter()
        with open(os.path.join(self.output_dir, "timer.txt"), "w") as f:
            f.write(str(self.end_time - self.start_time))
        logger.info(f"CODECARBON LOSS LOGGING: Rank {gpu_id} - Run {self.run_num} - Losses saved to {save_file_path}")
        self._log_gpu_util_stats(gpu_id, run_number)

    def _log_gpu_util_stats(self, gpu_id: int, run_number: str) -> None:
        phases = self.gpu_util_tracker._phases
        rows = []
        for phase in phases:
            util_hist = self.gpu_util_tracker.gpu_util[phase].history
            mem_hist  = self.gpu_util_tracker.mem_used[phase].history
            for i, (util, mem) in enumerate(zip(util_hist, mem_hist), start=1):
                rows.append(
                    {
                        "phase": phase,
                        "iteration": i,
                        "gpu_util_pct": util,
                        "mem_used_bytes": mem,
                        "cpu_util_pct": self.gpu_util_tracker.cpu_util[phase].history[i - 1],
                    }
                )
        if not rows:
            logger.warning("No GPU utilisation data collected — skipping CSV write.")
            return

        df = pd.DataFrame(rows)
        gpu_util_dir = os.path.join(self.output_dir, "gpu_util")
        os.makedirs(gpu_util_dir, exist_ok=True)
        save_path = os.path.join(
            gpu_util_dir, f"{run_number}gpu_util_rank_{gpu_id}.csv"
        )
        df.to_csv(save_path, index=False)
        logger.info(
            f"GPU UTIL LOGGING: Rank {gpu_id} - Run {self.run_num} - "
            f"GPU utilisation stats saved to {save_path}"
        )
        for phase in phases:
            util_hist = self.gpu_util_tracker.gpu_util[phase].history
            mem_hist  = self.gpu_util_tracker.mem_used[phase].history
            if not util_hist:
                continue
            avg_util = sum(util_hist) / len(util_hist)
            avg_mem  = sum(mem_hist)  / len(mem_hist)
            avg_cpu = sum(self.gpu_util_tracker.cpu_util[phase].history) / len(self.gpu_util_tracker.cpu_util[phase].history)
            logger.info(
                f"  [{phase:>10}] avg GPU util: {avg_util:6.2f}%  |  "
                f"avg mem used: {avg_mem / 1e9:.3f} GB  |  "
                f"avg CPU util: {avg_cpu:6.2f}%  (n={len(util_hist)})"
            )

    def log_loss(self, loss: torch.Tensor) -> None:
        """
        Take the loss from the training loop and log it to the CodeCarbon tracker file.
        """
        self.losses.append(
            {
                "task_name": f"Step #{self.iteration}",
                "loss": loss.to(torch.device("cpu"), non_blocking=True),
            }
        )


