# === import necessary modules ===
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision import transforms


"""
This file contains the code to train a ConvNext Large model using Simple trainer (src/trainer/simple.py).
It is based on the ConvNext Large model from PyTorch.
https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.convnext_large.html#convnext-large
"""

def build_transforms():
    """
    Build preprocessing transforms using pretrained weights metadata.
    """
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    return weights.transforms()


def init_convnext_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the ConvNext model.
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The ConvNext model.
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # This is a simple AdamW optimizer with weight decay. Choose different optimizers as needed.
    # Note: The learning rate is taken from the configuration object. Adjust it as needed for different models and training setups based on the loss function.
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)

def init_convnext_model(conf: config.Config, num_classes: int) -> nn.Module:
    """
    Initialize ConvNeXt-Large model.
    """
    os.environ["TORCH_HOME"] = 'home/slurm/comp597/students/andrewsaffar/weights'
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1

    model = convnext_large(weights=weights)

    # Replace classification head for custom dataset
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


################################################################################
#################################    Simple    #################################
################################################################################

def simple_trainer(conf : config.Config, model : nn.Module, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for ConvNext model. Uses the SimpleTrainer from src/trainer/simple.py.
    Args:
        conf (config.Config): The configuration object.
        model (transformers.GPT2LMHeadModel): The GPT-2 model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForLanguageModeling): The data collator to use.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
    """
    num_classes = 2
    loader = data.DataLoader(dataset, batch_size=conf.batch_size) # DataLoader for batching the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)    
    optimizer = init_convnext_optim(conf, model) # Initialize the optimizer for GPT-2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(loader),
    )
    loss_fn = nn.CrossEntropyLoss()
    # Return the SimpleTrainer with the initialized components
    return trainer.SimpleTrainer1(loader=loader, model=model, optimizer=optimizer, lr_scheduler=scheduler, device=device, stats=trainer_stats.init_from_conf(conf=conf, device=device, num_train_steps=len(loader)), loss_fn=loss_fn), None

################################################################################
##################################    Init    ##################################
################################################################################

def convnext_init(conf: config.Config, dataset: data.Dataset, num_classes: int) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the ConvNext model and returns the appropriate trainer based on the configuration.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
    """
    model = init_convnext_model(conf, num_classes)
    # Note: Currently, only Simple trainer is implemented for ConvNext. Add more trainers as needed.
    if conf.trainer == "simple": 
        return simple_trainer(conf, model, dataset)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")

