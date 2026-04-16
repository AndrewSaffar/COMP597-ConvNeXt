import tarfile
from pathlib import Path
from PIL import Image
import io
import torch
from torchvision.models import ConvNeXt_Large_Weights
import torch.utils.data as data
import csv
from PIL import Image
import random

data_load_name = "IF-CC95K"

class TarImageDataset(data.Dataset):
    def __init__(self, tar_path: str, metadata_csv: str, fraction: float = 1.0, seed: int = 42):
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r:gz")

        self.members = [
            m for m in self.tar.getmembers()
            if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        k = int(len(self.members) * fraction)
        self.members = self.members[:k]
        print("Number of images: ", k)
        self.label_lookup = {}
        with open(metadata_csv, newline="") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) < 2:
                    continue
                filename = Path(row[0]).name
                self.label_lookup[filename] =  int(row[1])

        # ConvNeXt transforms
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, idx: int):
        member = self.members[idx]

        file_obj = self.tar.extractfile(member)
        image = Image.open(io.BytesIO(file_obj.read())).convert("RGB")

        filename = Path(member.name).name

        if filename not in self.label_lookup:
            raise KeyError(f"Label not found for {filename}")
        label = self.label_lookup[filename]

        image = self.transform(image)

        return {
            "inputs": image,
            "targets": torch.tensor(label, dtype=torch.long),
        }
def load_data(conf: config.Config) -> data.Dataset:
    """
    Loads IF-CC95K directly from tar.gz without extracting.
    """
    metadata_csv = conf.fi_data_configs.metadata_path
    tar_path = conf.fi_data_configs.train_files

    return TarImageDataset(tar_path, metadata_csv, fraction=0.2)

