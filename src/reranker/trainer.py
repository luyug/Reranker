# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Tuple, Optional, Any, Union

from reranker.dist.sampler import SyncedSampler
from reranker.modeling import Reranker, RerankerDC

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer import Trainer, nested_detach
from transformers.trainer_utils import PredictionOutput, EvalPrediction
import logging

logger = logging.getLogger(__name__)


class RerankerTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _get_train_sampler(self):
        if self.args.local_rank == -1:
            return RandomSampler(self.train_dataset)
        elif self.args.collaborative:
            logger.info(f'Collaborative Mode.')
            return SyncedSampler(self.train_dataset, seed=self.args.seed)
        else:
            return DistributedSampler(self.train_dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        return super(RerankerTrainer, self).create_optimizer_and_scheduler(num_training_steps)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model: Reranker, inputs):
        return model(inputs)['loss']

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)

    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> PredictionOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label})

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

class RerankerDCTrainer(RerankerTrainer):
    def _chunk_input(self, inputs: Dict[str, torch.Tensor], chunk_size: int = None):
        if chunk_size is None:
            chunk_size = self.args.distance_cahce_stride
        keys = list(inputs.keys())
        for k, v in inputs.items():
            inputs[k] = v.split(chunk_size)

        chunks = []
        n_chunks = len(inputs[keys[0]])

        for i in range(n_chunks):
            chunks.append({k: inputs[k][i] for k in keys})

        return chunks

    def training_step(self, model: RerankerDC, inputs):
        model.train()
        _model = getattr(model, 'module', model)
        inputs = self._prepare_inputs(inputs)

        rnd_states = []
        all_logits = []
        chunks = self._chunk_input(inputs)

        for chunk in chunks:
            rnd_states.append(RandContext())
            if self.args.fp16:
                with torch.no_grad():
                    with autocast():
                        chunk_logits = model(chunk)
            else:
                with torch.no_grad():
                    chunk_logits = model(chunk)
            all_logits.append(chunk_logits)

        all_logits = torch.cat(all_logits).float()
        loss, grads = _model.compute_grad(all_logits)
        grads = grads.view(-1, self.args.distance_cahce_stride)

        for chunk_id, chunk in enumerate(chunks):
            with rnd_states[chunk_id]:
                if self.args.fp16:
                    with autocast():
                        surrogate = model(chunk, grads[chunk_id])
                else:
                    surrogate = model(chunk, grads[chunk_id])

            if self.args.gradient_accumulation_steps > 1:
                surrogate = surrogate / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        return loss.detach()



