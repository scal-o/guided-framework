from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Self

import click
import geopandas as gpd
import numpy as np
import openmatrix as omx
import pandas as pd
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from ml_static.transforms import SequentialTransform

if TYPE_CHECKING:
    from ml_static.config import Config


class DatasetSplit:
    """
    Manages train/val/test splits of a dataset with transforms.

    Handles splitting, transform fitting on training data, and provides
    convenient access to splits and indices.
    """

    def __init__(
        self,
        dataset: STADataset,
        split: tuple[float, float, float],
        transform: SequentialTransform | None = None,
        seed: int = 42,
    ):
        """
        Create dataset splits with optional transform.

        Args:
            dataset: Full dataset to split.
            split: Tuple of (train, val, test) proportions.
            transform: Transform to fit on training data and apply to all splits.
            seed: Random seed for reproducible splits.
        """
        self._dataset: STADataset = dataset
        self._split: tuple[float, float, float] = split

        # generate split indices
        self.indices: dict[str, list[int]] = self._create_split_indices(seed)

        # create dataset splits
        self.data_splits: dict[str, STADataset] = self._create_data_splits()

        # fit and inject transform if provided
        if transform is not None:
            self.transform: SequentialTransform = transform
            self._fit_transform()
            self._inject_transform()

    @classmethod
    def from_config(
        cls, config: Config, seed: int | None = None, force_reload: bool = False
    ) -> Self:
        """
        Create dataset split from configuration object.

        Args:
            config: Configuration object.
            force_reload: Whether to force dataset reload.

        Returns:
            DatasetSplit instance.
        """
        dataset: STADataset = STADataset.from_config(config, force_reload=force_reload)
        transform = SequentialTransform.from_config(config, stage="post")

        if seed is not None:
            d_seed = seed
        else:
            d_seed = config.training.seed if config.training.seed is not None else 42

        return cls(
            dataset=dataset,
            split=config.dataset.split,
            transform=transform,
            seed=d_seed,
        )

    def _create_split_indices(self, seed: int) -> dict[str, list[int]]:
        """
        Generate train/val/test indices.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Dictionary mapping split names to index lists.
        """
        # normalize split proportions
        split_total = sum(self._split)
        normalized = tuple(s / split_total for s in self._split)

        # calculate split sizes
        n_total = len(self._dataset)
        n_train = int(n_total * normalized[0])
        n_val = int(n_total * normalized[1])

        # generate shuffled indices
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)

        return {
            "train": indices[:n_train].tolist(),
            "val": indices[n_train : n_train + n_val].tolist(),
            "test": indices[n_train + n_val :].tolist(),
        }

    def _create_data_splits(self) -> dict[str, STADataset]:
        """
        Create dataset splits using computed indices.

        Returns:
            Dictionary mapping split names to dataset subsets.
        """
        # indices.items() always returns list[int] values, so indexing returns STADataset
        splits: dict[str, STADataset] = {}
        for split_name, indices in self.indices.items():
            subset = self._dataset[indices]
            assert isinstance(subset, STADataset), "Expected STADataset for list indexing"
            splits[split_name] = subset
        return splits

    def _fit_transform(self) -> None:
        """Fit transform on training split."""
        train_split = self.data_splits["train"]
        self.transform.fit_dataset(train_split)

    def _inject_transform(self) -> None:
        """Apply fitted transform to all splits."""
        for split in self.data_splits.values():
            split.transform = self.transform

    def __getitem__(self, key: str) -> STADataset:
        """
        Get dataset split by name.

        Args:
            key: One of 'train', 'val', 'test'.

        Returns:
            Requested dataset split.

        Raises:
            KeyError: If key is not valid.
        """
        if key not in self.data_splits:
            raise KeyError(f"Invalid split key '{key}'. Must be 'train', 'val', or 'test'.")
        return self.data_splits[key]


class STADataset(Dataset):
    """
    Pytorch Geometric dataset for static traffic assignment data.

    Each sample in the dataset is a graph representing the network state
    from a single scenario's static traffic assignment results.

    The node and link features closely resemble the ones presented in Liu & Meidani (2024).
    """

    def __init__(
        self,
        network_dir: Path | str,
        model: str = "default",
        max_nodes: int | None = None,
        **kwargs,
    ):
        """
        Args:
            network_dir (root) (str): Root directory where the dataset should be saved.
                        This directory is split into 'raw' and 'processed' subdirectories.
            model (str, optional): The model type for which the dataset is being prepared.
                        If "default", no pre-transforms are applied, and the dataset is created
                        as is.
            max_nodes (int, optional): Maximum number of nodes in the graph. Defaults to None.
            transform (callable, optional): A function/transform that takes in an
                                           `torch_geometric.data.Data` object and returns a
                                           transformed version. The data object will be
                                           transformed before every access. (default: None)
            pre_transform (callable, optional): A function/transform that takes in
                                                an `torch_geometric.data.Data` object and returns
                                                a transformed version. The data object will be
                                                transformed before being saved to disk.
                                                (default: None)
        """
        # create root directory for the dataset inside the network directory
        # i.e. alongside the other scenarios_* directories
        self.root = Path(network_dir) / "scenarios_sta_mldata"
        if not self.root.is_dir():
            self.root.mkdir()

        # define dirs
        self.network_dir = Path(network_dir)

        # save model type
        self.model = model
        self.max_nodes = max_nodes

        # define scenario names
        # retrieve all available scenarios and results
        scenarios_dir = self.network_dir / "scenarios_geojson"
        scenarios = list(scenarios_dir.glob("scenario_*"))
        scenarios = sorted(scenarios, key=lambda p: p.name)

        # remove scenario 0
        if (scenarios_dir / "scenario_00000").is_dir():
            scenarios.remove(scenarios_dir / "scenario_00000")

        self._scenario_paths: list[Path] = scenarios
        self._scenario_names: list[str] = [scenario.name for scenario in scenarios]

        # set default indices to include all scenarios
        self._indices = range(len(self._scenario_names))

        # init base class
        super().__init__(str(self.root), **kwargs)

    @classmethod
    def from_config(cls, config: Config, force_reload: bool = False) -> Self:
        """
        Create dataset from configuration object.

        Args:
            config: Configuration object.

        Returns:
            STADataset instance.
        """

        # extract model type
        model_name = config.model.type

        # create pre-transform from config
        pre_transform = SequentialTransform.from_config(config, stage="pre")

        # if force reload is set to false, check for pre-transform consistency and
        # force the reload anyway in case of mismatches
        if not force_reload:
            # path to the pre_transform repr
            pre_transform_path = (
                config.dataset.full_path
                / "scenarios_sta_mldata"
                / f"processed_{model_name}"
                / "pre_transform.pt"
            )

            # generate the current representation
            current_repr = str(pre_transform)

            # check for mismatch
            if pre_transform_path.exists():
                try:
                    saved_transform = torch.load(pre_transform_path, weights_only=False)
                    saved_repr = str(saved_transform)

                    if saved_repr != current_repr:
                        print(
                            f"Pre-transform change detected.\nSaved: {saved_repr}\nCurrent: {current_repr}\nTriggering reload..."
                        )
                        force_reload = True
                except Exception as e:
                    print(f"Could not load existing pre-transform: {e}. Triggering reload...")
                    force_reload = True

        # if force_reload is true, force reloading of the basedataset as well
        if force_reload:
            print("Forcing dataset reload...")
            BaseDataset.from_config(config, force_reload=True)

        # create dataset with model name and pre-transform
        dataset = cls(
            config.dataset.full_path,
            model=model_name,
            pre_transform=pre_transform,
            force_reload=force_reload,
            max_nodes=config.dataset.max_nodes,
        )

        return dataset

    ## ===========================
    ## === internal properties ===
    @property
    def scenario_paths(self) -> list[Path]:
        """List of scenario paths."""
        return [self._scenario_paths[i] for i in self.indices()]

    @property
    def scenario_names(self) -> list[str]:
        """List of scenario names."""
        return [self._scenario_names[i] for i in self.indices()]

    ## ====================================================
    ## === pytorch geometric dataset methods (required) ===
    @property
    def raw_dir(self) -> str:
        """
        Returns the directory where the raw data is stored.
        """
        return str(Path(self.root) / "preprocessed")

    @property
    def processed_dir(self) -> str:
        """
        Returns the directory where the processed data is stored.
        """
        return str(Path(self.root) / f"processed_{self.model}")

    @property
    def raw_file_names(self) -> list[str]:
        """
        Returns the list of files that must be present in the raw_dir directory
        in order to skip the download step.
        The file names are derived from the preprocessed scenarios.
        """
        scenario_names = self._scenario_names

        # to create a complete graph we need the scenario_xxxxx.pt file

        return [f"{scenario}.pt" for scenario in scenario_names]

    @property
    def processed_file_names(self) -> list[str]:
        """
        Return the list of files that must be present in the processed_dir directory
        in order to skip the processing step.
        The file names are derived from the scenarios available in the
        scenarios_sta_results directory.
        """
        scenario_names = self._scenario_names

        # we expect to find a single .pt file for every scenario
        return [f"{scenario}.pt" for scenario in scenario_names]

    def download(self):
        """
        The "download" consists in creaating the BaseDataset if not already present.
        This will automatically create the raw files needed for this dataset.
        """

        print("--- Dataset Download ---")
        BaseDataset(self.network_dir, force_reload=True)
        print("--- Dataset Download Complete ---")

    def process(self):
        """
        Processes raw data and saves it into the `processed_dir`.
        """

        print("--- Dataset Processing ---")
        print(f"Network: {self.network_dir.name}")
        print(f"Path: {self.network_dir}")

        scenario_names = self._scenario_names

        for scenario in tqdm(scenario_names, desc="Processing data"):
            # raw dir
            src = Path(self.raw_dir)

            # load base data
            data = torch.load(src / f"{scenario}.pt", weights_only=False)

            # set num_nodes: this is a pytorch geometric requirement,
            # as some transforms make it unable to determine it dynamically
            data["nodes"].num_nodes = data["_raw"].real_index.max().item() + 1
            data["nodes"].max_nodes = self.max_nodes

            # apply pre-transform if defined
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            try:
                data.validate()
            except AssertionError as e:
                raise AssertionError(f"Data validation failed: {e}") from e

            torch.save(data, Path(self.processed_dir) / f"{scenario}.pt")

        print("--- Dataset Processing Complete ---")

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        scenario = self._scenario_names[idx]
        data = torch.load(Path(self.processed_dir) / f"{scenario}.pt", weights_only=False)
        return data

    def __getitem__(self, idx) -> STADataset | HeteroData:  # type: ignore
        """
        Handles indexing and slicing of the dataset.
        If an integer is passed, it returns a single data object.
        If a slice or list of indices is passed, it returns a `Subset` object
        with the sliced custom attributes attached.
        """
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        # return a subset of the original dataset if a list or slice is passed
        subset = copy.copy(self)

        indices = subset.indices()
        subset._indices = sorted([indices[i] for i in idx])

        return subset

    def __iter__(self) -> Iterator[HeteroData]:
        for i in range(len(self)):
            yield self[i]


class BaseDataset(Dataset):
    """
    Pytorch Geometric dataset for static traffic assignment data.

    The BaseDataset serves as the first processing step of the raw
    traffic assignment data, converting it into standardized Pytorch Geometric
    Data objects.

    Each sample in the dataset is a graph representing the network state
    from a single scenario's static traffic assignment results.
    """

    def __init__(self, network_dir: Path | str, force_reload: bool = False):
        """
        Args:
            network_dir (root) (str): Root directory where the dataset should be saved.
                        This directory is split into 'raw' and 'processed' subdirectories.
        """
        # create root directory for the dataset inside the network directory
        # i.e. alongside the other scenarios_* directories
        self.root = Path(network_dir) / "scenarios_sta_mldata"
        if not self.root.is_dir():
            self.root.mkdir()

        # define dirs
        self.network_dir = Path(network_dir)
        self.node_data_dir = self.network_dir / "scenarios_geojson"
        self.link_data_dir = self.network_dir / "scenarios_sta_results"

        # define scenario names
        # retrieve all available scenarios and results
        scenarios = list(self.node_data_dir.glob("scenario_*"))
        results = list(self.link_data_dir.glob("scenario_*"))

        # remove scenario 0 from both scenario and results
        if (self.node_data_dir / "scenario_00000").is_dir():
            scenarios.remove(self.node_data_dir / "scenario_00000")
        if (self.link_data_dir / "scenario_00000").is_dir():
            results.remove(self.link_data_dir / "scenario_00000")

        # sort paths deterministically by name to ensure reproducible ordering
        # across different systems and filesystem implementations
        scenarios = sorted(scenarios, key=lambda p: p.name)
        results = sorted(results, key=lambda p: p.name)

        # check there are the same number of scenarios and results
        if len(scenarios) != len(results):
            raise ValueError(
                f"Found {len(scenarios)} scenario files but {len(results)} results files."
            )

        self._scenario_paths: list[Path] = scenarios
        self._scenario_results: list[Path] = results
        self._scenario_names: list[str] = [scenario.name for scenario in scenarios]

        # set default indices to include all scenarios
        self._indices = range(len(self._scenario_names))

        # init base class
        super().__init__(str(self.root), force_reload=force_reload)

    @classmethod
    def from_config(cls, config: Config, force_reload: bool = False) -> Self:
        """
        Create dataset from configuration object.

        Args:
            data_path: Path to the network data directory.
            config: Configuration object.

        Returns:
            STADataset instance.
        """
        # create dataset
        dataset = cls(config.dataset.full_path, force_reload=force_reload)

        return dataset

    ## ===========================
    ## === internal properties ===
    @property
    def scenario_paths(self) -> list[Path]:
        """List of scenario paths."""
        return [self._scenario_paths[i] for i in self.indices()]

    @property
    def scenario_results(self) -> list[Path]:
        """List of scenario result paths."""
        return [self._scenario_results[i] for i in self.indices()]

    @property
    def scenario_names(self) -> list[str]:
        """List of scenario names."""
        return [self._scenario_names[i] for i in self.indices()]

    ## ====================================================
    ## === pytorch geometric dataset methods (required) ===
    @property
    def processed_dir(self) -> str:
        """
        Returns the directory where the processed data is stored.
        """
        return str(Path(self.root) / "preprocessed")

    @property
    def raw_file_names(self) -> list[str]:
        """
        Returns the list of files that must be present in the raw_dir directory
        in order to skip the download step.
        The file names are derived from the scenarios available in the
        scenarios_sta_results directory.
        """
        scenario_names = self._scenario_paths

        # To create a complete graph we need:
        # - OD demand matrix (.omx)
        # - Physical static traffic assignment link results (.parquet)
        names_od = [f"{scenario.name}_od.omx" for scenario in scenario_names]
        names_links = [f"{scenario.name}.parquet" for scenario in scenario_names]

        return names_od + names_links

    @property
    def processed_file_names(self) -> list[str]:
        """
        Return the list of files that must be present in the processed_dir directory
        in order to skip the processing step.
        """
        scenario_names = self._scenario_paths
        return [f"{scenario.name}.pt" for scenario in scenario_names]

    def download(self):
        """
        Copies OD matrix files (.omx) and assignment result files (.parquet)
        to the dataset raw directory with standardized file naming.
        """

        print("--- Dataset Download ---")
        print(f"Network: {self.network_dir.name}")
        print(f"Path: {self.network_dir}")

        scenario_paths = self._scenario_paths
        result_paths = self._scenario_results

        dest = Path(self.raw_dir)

        for scenario in tqdm(scenario_paths, desc="Copying OD demand data"):
            shutil.copy(scenario / "od.omx", dest / f"{scenario.name}_od.omx")

        for scenario in tqdm(result_paths, desc="Copying link assignment results"):
            shutil.copy(
                scenario / f"{scenario.name}.parquet",
                dest / f"{scenario.name}.parquet",
            )

        print("--- Dataset Download Complete ---")

    def process(self):
        """
        Processes raw scenario data and converts it into PyTorch Geometric HeteroData graphs saved to `processed_dir`.
        """

        print("--- Base Dataset Processing ---")
        print(f"Network: {self.network_dir.name}")
        print(f"Path: {self.network_dir}")

        scenario_names = self._scenario_names

        for scenario in tqdm(scenario_names, desc="Processing data"):
            src = Path(self.raw_dir)

            ## 1. Physical (Real) Network Links & Topology Node Count
            # Read static traffic assignment equilibrium results from Parquet
            link_df = pd.read_parquet(src / f"{scenario}.parquet")

            # Create 0-based physical link edge index tensor [2, E_real]
            real_edges = link_df[["a_node", "b_node"]].values
            real_edges -= 1  # Convert 1-based node IDs to 0-based array indices

            # Physical node count is determined by maximum node ID referenced in topology links
            num_nodes = int(real_edges.max()) + 1

            ## 2. Nodes & Origin-Destination (OD) Demand Matrix
            # Open binary OpenMatrix (.omx) demand file
            omx_mat = omx.open_file(src / f"{scenario}_od.omx")
            od_mat = np.array(omx_mat["matrix"])

            # Scale down daily OD demand matrix to peak-hour trips (peak hour = ~10% of daily demand)
            od_mat = od_mat / 10.0

            # Expand centroid-only OMX matrix to full [N_nodes x N_nodes] matrix with zero-padding for non-centroid nodes
            full_mat = pd.DataFrame(od_mat)
            full_mat = full_mat.reindex(
                index=range(0, num_nodes),
                columns=range(0, num_nodes),
                fill_value=0.0,
            )
            full_mat = full_mat.values
            omx_mat.close()

            # Convert OD matrix to PyTorch float tensor
            od_mat = torch.as_tensor(full_mat, dtype=torch.float)

            # Extract link capacity and flow attributes; scale capacity and flow to peak hour (/ 10.0)
            edge_capacity = link_df["capacity"].values / 10.0
            edge_free_flow_time = link_df["free_flow_time"].values
            edge_vcr = link_df["volume_capacity_ratio"].values
            edge_flow = link_df["flow"].values / 10.0

            # Convert physical link features to PyTorch tensors
            real_edges = torch.as_tensor(real_edges, dtype=torch.long).t()
            edge_capacity = torch.as_tensor(edge_capacity, dtype=torch.float)
            edge_free_flow_time = torch.as_tensor(edge_free_flow_time, dtype=torch.float)
            edge_vcr = torch.as_tensor(edge_vcr, dtype=torch.float)
            edge_flow = torch.as_tensor(edge_flow, dtype=torch.float)

            ## 3. Virtual (OD Pair) Links
            # Construct virtual edges for all OD pairs with positive travel demand (d_q > 0)
            indices = np.where(full_mat > 0)
            virtual_edges = torch.as_tensor(np.array(indices), dtype=torch.long)

            ## 4. Heterogeneous Graph Assembly
            data = HeteroData()

            # Populate temporary _raw namespace for downstream builder transforms to access
            data["_raw"].demand = od_mat
            data["_raw"].real_index = real_edges
            data["_raw"].virtual_index = virtual_edges
            data["_raw"].edge_capacity = edge_capacity
            data["_raw"].edge_free_flow_time = edge_free_flow_time
            data["_raw"].edge_vcr = edge_vcr
            data["_raw"].edge_flow = edge_flow

            torch.save(data, Path(self.processed_dir) / f"{scenario}.pt")

        print("--- Dataset Processing Complete ---")

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        scenario = self._scenario_names[idx]
        data = torch.load(Path(self.processed_dir) / f"{scenario}.pt", weights_only=False)
        return data

    def __getitem__(self, idx) -> HeteroData:  # type: ignore
        """
        Handles indexing and slicing of the dataset.
        If an integer is passed, it returns a single data object.
        If a slice or list of indices is passed, it returns a `Subset` object
        with the sliced custom attributes attached.
        """
        if isinstance(idx, int):
            return self.get(self.indices()[idx])
        else:
            raise NotImplementedError("Slicing is not implemented for BaseDataset.")


class InfiniteBatchSampler(Sampler):
    """
    A PyTorch Sampler that generates an infinite stream of batches from a dataset.
    This allows for the background dataloader workers to continuously fetch data without waiting
    for the main training loop, reducing idle time between epochs.

    Args:
        data_source (Dataset): The dataset to sample from.
        batch_size (int): The number of samples in each batch.
        shuffle (bool): Whether to shuffle the dataset indices before sampling.
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # create a separate random generator for the sampler
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def __iter__(self) -> Iterator[list[int]]:

        # get length of the dataset
        n = len(self.data_source)

        # while True to avoid sending StopIteration signals to the dataloader workers
        while True:
            # generate a list of indices
            indices = list(range(n))

            if self.shuffle:
                indices = torch.randperm(n, generator=self.generator).tolist()

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:

        n = len(self.data_source)

        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size


@click.command()
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the networks directory.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-creation of the dataset.",
)
def create_dataset(
    network: str,
    path: Path | str | None = None,
    force: bool = False,
):
    """
    Creates a Pytorch Geometric dataset from NETWORK's static traffic assignment data.
    """
    print("--- Dataset Creation ---")
    print(f"Network: {network}")
    print(f"Base path: {path}")

    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    network_path = path / network

    if not network_path.is_dir():
        raise ValueError(
            f"Network path {path} does not exist."
            "Make sure to run the script from the network directory or provide a base_path."
        )

    try:
        # initialize the dataset
        STADataset(network_path, force_reload=force)

        print("--- Dataset Creation Complete ---")

    except Exception as e:
        raise Exception(f"Dataset generation failed. An unexpected error occurred: {e}") from e
