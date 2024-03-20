import os
from typing import Union, List, Tuple, Optional, Callable

import torch
from pydgn.data.dataset import DatasetInterface
from torch_geometric.data import Data


class GraphPropertyPrediction(DatasetInterface):
    def __init__(
        self,
        root: str,
        name: str,
        dim="25-35",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        TASKS = ["dist", "ecc", "diam"]

        self.name_file_dict = {
            "SSSP": "dist",
            "Diameter": "diam",
            "Eccentricity": "ecc",
        }

        assert transform == None
        assert pre_transform == None
        assert pre_filter == None

        super().__init__(root, name, transform, pre_transform, pre_filter)

        assert self.name_file_dict[self.name] in TASKS
        assert dim in ["25-35"]

        # target has already been normalized by authors of ADGN paper
        self.data_list = torch.load(self.raw_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        return [f"{self.name_file_dict[self.name]}_25-35_data_list.pt"]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_file_names

    def download(self):
        raise NotImplementedError(
            "you should already provide the raw files "
            f"in the folder {self.root}/{self.name}/raw"
        )

    def process(self):
        pass

    def get(self, idx: int) -> Data:
        r"""
        Gets the data object at index :obj:`idx`.
        """
        return self.data_list[idx]

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.data_list[0].x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return 1

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self) -> int:
        return len(self.data_list)


class Peptides(DatasetInterface):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.name = name
        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []  # you should already have processed this

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).edge_attr.shape[1]

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.get(0).y.shape[1]

    def get(self, idx: int) -> Data:
        d = self.data[idx]
        d.x = d.x.float()
        return d

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self):
        return len(self.data)
