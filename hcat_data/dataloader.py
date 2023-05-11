import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import glob
import os.path
import skimage.io as io
import xml

from typing import *


def get_dtype_offset(dtype: str = "uint16", image_max=None) -> int:
    """get dtype from string"""

    encoding = {
        "uint16": 2**16,
        "uint8": 2**8,
        "uint12": 2**12,
        "float64": 1,
    }
    dtype = str(dtype)
    if dtype in encoding:
        scale = encoding[dtype]
    else:
        print(
            f"\x1b[1;31;40m"
            + f"ERROR: Unsupported dtype: {dtype}. Currently support: {[k for k in encoding]}"
            + "\x1b[0m"
        )
        scale = None
        if image_max:
            print(f"Appropriate Scale Factor inferred from image maximum: {image_max}")
            if image_max <= 256:
                scale = 256
            else:
                scale = image_max
    return scale


class dataset(Dataset):
    def __init__(
        self,
        path: Union[List[str], str],
        transforms: Callable = lambda x: x,
        pad_size: int = 100,
        sample_per_image: int = 1,
        metadata: Optional[Dict[str, str]] = None,
    ):
        super(Dataset, self).__init__()

        # Reassigning variables
        self.mask = []
        self.image = []
        self.centroids = []
        self.boxes = []
        self.labels = []
        self.files = []
        self.transforms = transforms
        self.pad_size: List[int] = [pad_size, pad_size]

        self.metadata = metadata

        self.device = "cpu"

        self.sample_per_image = sample_per_image

        path: List[str] = [path] if isinstance(path, str) else path

        # Find only files
        for p in path:
            self.files.extend(glob.glob(f"{p}{os.sep}*.xml"))

        for f in self.files:
            if os.path.exists(f[:-4:] + ".tif"):
                image_path = f[:-4:] + ".tif"
            elif os.path.exists(f[:-4:] + ".png"):
                image_path = f[:-4:] + ".png"
            else:
                raise FileNotFoundError(
                    f"Could not find file: {image_path[:-4:]} with extensions .tif or .png"
                )

            image: np.array = io.imread(image_path)

            image = image.transpose(2, 0, 1) if image.shape[-1] <= 3 else image
            scale: int = get_dtype_offset(image.dtype)
            if not scale:
                print(f, image.max(), image.shape)
            image: Tensor = TF.pad(
                torch.from_numpy(image / scale), self.pad_size
            ).unsqueeze(-1)

            boxes, classes = self.data_from_xml(f)

            self.image.append(image.half().to(memory_format=torch.channels_last))

            b = torch.tensor(boxes) + pad_size
            c = torch.tensor(classes)

            b = b[c < 10, :]
            c = c[c < 10]

            self.boxes.append(b)
            self.labels.append(c)

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Tuple[str, Tensor]:
        item = (
            item // self.sample_per_image
        )  # We might artificailly want to sample more times per image...

        with torch.no_grad():
            data_dict = {
                "image": self.image[item].squeeze(-1),
                "boxes": self.boxes[item],
                "labels": self.labels[item],
            }

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        return data_dict

    def get_file_at_index(self, item):
        return self.files[item // self.sample_per_image]

    def get_labels_at_index(self, item):
        return self.labels[item // self.sample_per_image]

    def to(self, device: str):
        self.device = device
        self.image = [x.to(device) for x in self.image]
        self.boxes = [x.to(device) for x in self.boxes]
        self.labels = [x.to(device) for x in self.labels]

        return self

    def cuda(self):
        self.device = "cuda:0"
        return self.to("cuda:0")

    def cpu(self):
        self.device = "cpu"
        return self.to("cpu")


    @staticmethod
    def prepare_image(image: Tensor, device: str = "cpu"):
        c, x, y = image.shape

        for i in range(c):
            max = image[i, ...].max()
            max = max if max != 0 else 1
            image[i, ...] = image[i, ...].div(max)

        if c < 3:
            image = torch.cat(
                (torch.zeros((1, x, y), device=image.device), image), dim=0
            )

        return image.to(device).float()

    @staticmethod
    def data_from_xml(f: str) -> Tuple[List[List[int]], List[int]]:
        """
        Parse a XML data file from labelImg to the raw box annotatiosn

        :param f: xml file
        :return: List -> Boxes, Labels
        """
        root = xml.etree.ElementTree.parse(f).getroot()

        boxes: List[List[int]] = []
        labels: List[int] = []
        for c in root.iter("object"):
            for box, cls in zip(c.iter("bndbox"), c.iter("name")):
                label: str = cls.text

                if label in ["OHC", "IHC"]:
                    label: int = 1 if label == "OHC" else 2
                    box = [
                        int(box.find(s).text) for s in ["xmin", "ymin", "xmax", "ymax"]
                    ]

                    boxes.append(box)
                    labels.append(label)

        return boxes, labels


class MultiDataset(Dataset):
    def __init__(self, *args):
        self.datasets: List[Dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]
        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self):
        return len(self._mapped_indicies)

    def __getitem__(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i][item - _offset]
        except RuntimeError:
            print(
                i,
                _offset,
                item - _offset,
                item,
                len(self.datasets[i]),
                self.datasets[i].files[item],
            )
            raise RuntimeError

    def get_file_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i].get_file_at_index(item - _offset)
        except IndexError:
            print(
                i,
                _offset,
                item - _offset,
                item,
                len(self.datasets[i]),
                self.datasets[i],
            )
            raise RuntimeError

    def get_cells_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        labels = self.datasets[i].get_labels_at_index(item - _offset)
        ohc = sum([1 for l in labels if l == 1])
        ihc = sum([1 for l in labels if l == 2])

        return ohc, ihc

    def get_metadata_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i].metadata
        except IndexError:
            print(
                i,
                _offset,
                item - _offset,
                item,
                len(self.datasets[i]),
                self.datasets[i],
            )
            raise RuntimeError

    def to(self, device: str):
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self):
        for i in range(self.num_datasets):
            self.datasets[i].to("cuda:0")
        return self

    def cpu(self):
        for i in range(self.num_datasets):
            self.datasets[i].to("cpu")
        return self


def colate(
    data_dict: List[Dict[str, Tensor]]
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images = [dd.pop("image").squeeze(-1) for dd in data_dict]

    return images, data_dict
