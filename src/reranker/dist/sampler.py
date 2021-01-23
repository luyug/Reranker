# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch.utils.data import DistributedSampler, Dataset

import logging
logger = logging.getLogger(__name__)


class SyncedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        super(SyncedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed)
        self.num_samples = len(self.dataset)
        self.total_size = len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # DO NOT SUB SAMPLE!
        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int):
        super(SyncedSampler, self).set_epoch(epoch)
        logger.info(f'Setting Data Sampler Epoch to {epoch}')
