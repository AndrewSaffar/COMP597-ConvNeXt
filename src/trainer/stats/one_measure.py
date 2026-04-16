import time

from codecarbon import track_emissions, EmissionsTracker, OfflineEmissionsTracker
from codecarbon.core.util import backup
from codecarbon.external.logger import logger
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData
from typing import List
import codecarbon
import codecarbon.core.cpu 
import logging
import os
import csv
import pandas as pd
import src.config as config
import src.trainer.stats.base as base
import torch
import pynvml
from src.trainer.stats.utils import RunningStat

logger = logging.getLogger(__name__)

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name="one_measure"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon trainer stats. Using default PyTorch device")
        device = torch.get_default_device() 
    return CodeCarbonStats(device, conf.trainer_stats_configs.codecarbon.run_num, conf.trainer_stats_configs.codecarbon.project_name, conf.trainer_stats_configs.codecarbon.output_dir)

class SimpleFileOutput(BaseOutput): 
    
    def __init__(self, 
        output_file_name: str = "codecarbon.csv", 
        output_dir: str = ".",
        on_csv_write: str = "append"
    ):
        if on_csv_write not in {"append", "update"}:
            raise ValueError(
                f"Unknown `on_csv_write` value: {on_csv_write}"
                + " (should be one of 'append' or 'update'"
            )
        
        self.output_file_name: str = output_file_name
        
        if not os.path.exists(output_dir):
            raise OSError(f"Folder '{output_dir}' doesn't exist !")
        
        self.output_dir: str = output_dir
        self.on_csv_write: str = on_csv_write
        self.save_file_path = os.path.join(self.output_dir, self.output_file_name) #default: ./codecarbon.csv
        
        logger.info(
        f"Emissions data (if any) will be saved to file {os.path.abspath(self.save_file_path)}"
        )

    def has_valid_headers(self, data: EmissionsData):
        with open(self.save_file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dict_from_csv = dict(list(csv_reader)[0])
            list_of_column_names = list(dict_from_csv.keys())
            return list(data.values.keys()) == list_of_column_names

    def to_csv(self, total: EmissionsData, delta: EmissionsData):
        """
        Save the emissions data to a CSV file.
        If the file already exists, append the new data to it.
        param `delta` is not used in this method.
        """

        # Add code to check whether a part of the save_file_path already exists -->
        # in our case, the output_file_name-experiment_name file exists, but the current code
        # is only checking whether output_file_name exists. Since it doesn't, it goes ahead 
        # and creates this new file, despite it being a "useless" file.

        # Problem: stop() calls persist_data() which calls out() and task_out(). Thus the task csv file is accurate.
        # But is out() accurate with tasks? How do tasks update total_emissions and delta? 

        file_exists: bool = os.path.isfile(self.save_file_path)
        
        if file_exists and not self.has_valid_headers(total): # CSV headers changed
            logger.warning("The CSV format have changed, backing up old emission file.")
            backup(self.save_file_path)
            file_exists = False 
        
        new_df = pd.DataFrame.from_records([dict(total.values)])

        if not file_exists:
            df = new_df
        elif self.on_csv_write == "append":
            df = pd.read_csv(self.save_file_path)
            df = pd.concat([df, new_df])
        else:
            df = pd.read_csv(self.save_file_path)
            df_run = df.loc[df.run_id == total.run_id]
            if len(df_run) < 1:
                df = pd.concat([df, new_df])
            elif len(df_run) > 1:
                logger.warning(
                f"CSV contains more than 1 ({len(df_run)})"
                + f" rows with current run ID ({total.run_id})."
                + "Appending instead of updating."
                )
                df = pd.concat([df, new_df])
            else:
                df.loc[df.run_id == total.run_id, list(total.values.keys())] = list(total.values.values())
    
        df.to_csv(self.save_file_path, index=False)

    def out(self, total: EmissionsData, delta: EmissionsData):
        self.to_csv(total, delta)

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        pass

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str):
        # run_id = data[0].run_id
        split = os.path.splitext(self.save_file_path)
        save_task_file_path = split[0] + "-" + experiment_name + split[1]
        df = pd.DataFrame(columns=data[0].values.keys())
        new_df = pd.DataFrame.from_records(
            [dict(data_point.values) for data_point in data]
        )
        # Filter out empty or all-NA columns, to avoid warnings from Pandas
        new_df = new_df.dropna(axis=1, how="all")
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(save_task_file_path, index=False)

class GpuUtilisationStat:

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._phases = ["step", "forward", "backward", "optimizer"]
        self.gpu_util: dict[str, RunningStat] = {p: RunningStat() for p in self._phases}
        self.mem_used: dict[str, RunningStat] = {p: RunningStat() for p in self._phases}

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
        
        # Normal-mode tracker to track the entire training loop
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name = project_name, 
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False, 
            output_handlers = [SimpleFileOutput(output_file_name = f"{run_number}cc_full_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            log_level = "warning",
            gpu_ids = [gpu_id],
        )        
    

    def start_train(self) -> None:
        self.start_time = time.perf_counter()
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.start()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.stop()
        self.end_time = time.perf_counter()

    def start_step(self) -> None:
        pass
        
    def stop_step(self) -> None:
        pass

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
        pass

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

        logger.info(f"CODECARBON LOSS LOGGING: Rank {gpu_id} - Run {self.run_num} - Losses saved to {save_file_path}")
        self._log_gpu_util_stats(gpu_id, run_number)
        with open(os.path.join(self.output_dir, "timer.txt"), "w") as f:
            f.write(str(self.end_time - self.start_time))

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
            logger.info(
                f"  [{phase:>10}] avg GPU util: {avg_util:6.2f}%  |  "
                f"avg mem used: {avg_mem / 1e9:.3f} GB  (n={len(util_hist)})"
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


