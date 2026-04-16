# === import necessary modules ===
from src.models.convnext.convnext import convnext_init
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "convnext"

def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return convnext_init(conf, dataset, num_classes=2)
