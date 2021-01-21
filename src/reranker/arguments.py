# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    temperature: Optional[float] = field(default=None)


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)
    dev_path: str = field(
        default=None, metadata={"help": "Path to dev data"}
    )
    pred_path: List[str] = field(default=None, metadata={"help": "Path to prediction data"})
    pred_dir: str = field(
        default=None, metadata={"help": "Path to prediction directory"}
    )
    pred_id_file: str = field(default=None)
    rank_score_path: str = field(default=None, metadata={"help": "where to save the match score"})
    max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]
        if self.pred_dir is not None:
            files = os.listdir(self.pred_dir)
            self.pred_path = [
                os.path.join(self.pred_dir, f)
                for f in files
            ]


@dataclass
class RerankerTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    distance_cahce: bool = field(default=False)
    distance_cahce_stride: int = field(default=2)

    collaborative: bool = field(default=False)
