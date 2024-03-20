from random import shuffle

import numpy as np
import pydgn
import torch
from numpy import random
from pydgn.data.splitter import Splitter, OuterFold, InnerFold


class GraphPropPredSplitter(Splitter):
    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        assert len(dataset) == 5120 + 640 + 1280
        assert self.n_inner_folds == 1
        self.n_outer_folds == 1

        train_idxs = torch.arange(0, 5120)
        val_idxs = torch.arange(5120, 5120 + 640)
        test_idxs = torch.arange(5120 + 640, 5120 + 640 + 1280)

        inner_fold_splits = []

        inner_fold = InnerFold(
            train_idxs=train_idxs.tolist(),
            val_idxs=val_idxs.tolist(),
        )
        inner_fold_splits.append(inner_fold)

        self.inner_folds.append(inner_fold_splits)

        outer_fold = OuterFold(
            train_idxs=train_idxs.tolist(),
            val_idxs=val_idxs.tolist(),
            test_idxs=test_idxs.tolist(),
        )

        self.outer_folds.append(outer_fold)


class PeptidesSplitter(Splitter):
    # PyG version
    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        idxs = list(range(len(dataset)))
        shuffle(idxs)

        assert len(dataset) == 10873 + 2331 + 2331
        assert self.n_inner_folds == 1
        self.n_outer_folds == 1

        train_idxs = idxs[:10873]
        val_idxs = idxs[10873 : 110831 + 2331]
        test_idxs = idxs[10873 + 2331 : 110831 + 2331 + 2331]

        inner_fold_splits = []

        inner_fold = InnerFold(
            train_idxs=train_idxs,
            val_idxs=val_idxs,
        )
        inner_fold_splits.append(inner_fold)

        self.inner_folds.append(inner_fold_splits)

        outer_fold = OuterFold(
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs,
        )

        self.outer_folds.append(outer_fold)
