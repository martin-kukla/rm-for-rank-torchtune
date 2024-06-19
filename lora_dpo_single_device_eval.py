# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.peft.peft_utils import (
    disable_adapter,
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.recipe_interfaces import EvalRecipeInterface
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class LoRADPORecipeSingleDeviceEval(EvalRecipeInterface):
    """
    LoRA DPO recipe for dense transformer-based LLMs such as Llama2 for
    single device training. This is based on HF's DPOTrainer in the
    TRL library: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L65

    This recipe supports:
        - Full bf16 training for supported HW architectures. We currently check bf16 support via
        the `torch.cuda.is_bf16_supported` API. This is disabled by default but can be enabled via
        setting `dtype=bf16` in configuration.
        - Logging to terminal, WandB, or TensorBoard.

    Assumptions:
        - Datasets are Map-style and data fits in memory (not streamed).

    The following configs can be used to run this recipe:
        >>> tune ls
        RECIPE                          CONFIG
        lora_dpo_single_device          llama2/7B_lora_dpo_single_device

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir

        self.seed = utils.set_seed(seed=cfg.seed) # TODO XXX XXX: We shouldn't need this
        self.max_eval_steps = cfg.max_eval_steps


    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
            lora_weights_state_dict = (checkpoint_dict[utils.ADAPTER_KEY]),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )


    def _setup_model(
        self,
        cfg_model: DictConfig,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.adapter_params = get_adapter_params(model)
        #set_trainable_params(model, self.adapter_params) # TODO XXX

        validate_state_dict_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=cfg_model.apply_lora_to_output,
            full_model_state_dict_keys=model.state_dict().keys(),
            lora_state_dict_keys=(
                lora_weights_state_dict.keys()
                if lora_weights_state_dict is not None
                else None
            ),
            base_model_state_dict_keys=base_model_state_dict.keys(),
        )

        model.load_state_dict(base_model_state_dict, strict=False)
        if lora_weights_state_dict:
            model.load_state_dict(lora_weights_state_dict, strict=False)

        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        utils.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        log.info(f"Model is initialized with precision {self._dtype}.")
        if self._device == torch.device("cuda"):
            memory_stats = utils.memory_stats_log(device=self._device)
            log.info(f"Memory Stats after model init:\n{memory_stats}")
        return model

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
        )
        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate_dpo,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
            ),
        )
        log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def concatenated_forward(
        self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass of the model with chosen and rejected samples concatenated.

        Args:
            model (nn.Module): The model to be used for the forward pass.
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of input_ids and labels.

        Returns:
            Tuple of chosen log probs, rejected log probs, chosen logits, rejected logits.
        """
        concatenated_input_ids, concatenated_labels = batch
        concatenated_input_ids = concatenated_input_ids.to(self._device)
        concatenated_labels = concatenated_labels.to(self._device)

        # formed by concatenating an equal number of "chosen" and "rejected".
        len_chosen = concatenated_input_ids.shape[0] // 2

        all_logits = model(concatenated_input_ids)

        all_log_probs = self.get_batch_log_probs(all_logits, concatenated_labels)

        chosen_log_probs = all_log_probs[:len_chosen]
        rejected_log_probs = all_log_probs[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits)

    @staticmethod
    def get_batch_log_probs(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = CROSS_ENTROPY_IGNORE_IDX,
    ) -> torch.FloatTensor:
        """
        Calculate log probabilities based on provided logits and labels.

        Args:
            logits (torch.FloatTensor): direct logits output of the model of shape (b, s, v)
            labels (torch.LongTensor): ground-truth labels to compute log probs with, shape (b, s).
                Label tokens with a value of label_pad_token_id are ignored.
            label_pad_token_id (int): token id to ignore in labels.

        Returns:
            Calculated log probs of shape (b, )

        Raises:
            ValueError: If logits and labels have different shapes.
        """

        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0
        # take log-likelihood of the labels given our model
        per_token_log_probs = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_log_probs * loss_mask).sum(-1)

    def evaluate(self) -> None:
        """
        The core evaluation loop.
        """
        steps_metrics = []

        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        self._sampler.set_epoch(0) # TODO XXX XXX: I don't need to shuffle for eval. For now, I test on small portion of dataset which small training run has seen
        for eval_step, batch in enumerate(pbar := tqdm(self._dataloader)):
            if (
                self.max_eval_steps is not None
                and eval_step == self.max_eval_steps
            ):
                break

            with torch.no_grad():
                (
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    policy_chosen_logits,
                    policy_rejected_logits,
                ) = self.concatenated_forward(self._model, batch)

                with disable_adapter(self._model):
                    (
                        reference_chosen_log_probs,
                        reference_rejected_log_probs,
                        _,
                        _,
                    ) = self.concatenated_forward(self._model, batch)

                loss, chosen_rewards, rejected_rewards = self._loss_fn(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    reference_chosen_log_probs,
                    reference_rejected_log_probs,
                )
                loss = loss.mean()
                reward_accuracies = (chosen_rewards > rejected_rewards).float()
    
                # TODO XXX: Debugging
                #print(f'loss {loss} reward_accuracies {reward_accuracies} policy_chosen_log_probs {policy_chosen_log_probs} reference_chosen_log_probs {reference_chosen_log_probs} batch {batch}')
            
            pbar.set_description(f"{eval_step+1}|Loss: {loss.item()}")
            self._metric_logger.log_dict(
                {
                    "loss": loss.item(),
                    "rewards/chosen": chosen_rewards.mean().cpu(),
                    "rewards/rejected": rejected_rewards.mean().cpu(),
                    "rewards/accuracies": reward_accuracies.mean().cpu(),
                    "rewards/margins": (chosen_rewards - rejected_rewards)
                    .mean()
                    .cpu(),
                    "log_probs/rejected": policy_rejected_log_probs.detach()
                    .mean()
                    .cpu(),
                    "log_probs/chosen": policy_chosen_log_probs.detach()
                    .mean()
                    .cpu(),
                    "logits/rejected": policy_rejected_logits.detach()
                    .mean()
                    .cpu(),
                    "logits/chosen": policy_chosen_logits.detach().mean().cpu(),
                },
                step=eval_step,
            )
            steps_metrics.append((loss.item(), chosen_rewards.mean().cpu(), rejected_rewards.mean().cpu(), reward_accuracies.mean().cpu(), (chosen_rewards - rejected_rewards).mean().cpu()))

        # Aggregated evaluation metrics
        steps_metrics = list(zip(*steps_metrics))
        metrics_names = ["Loss", "Chosen rewards", "Rejected reward", "Reward accuracies", "Margins"]
        print(f'Steps metrics: {steps_metrics}')
        my_mean = lambda x: sum(x)/len(x)
        for m_name, m_numbers in zip(metrics_names, steps_metrics):
            print(f'{m_name}: {my_mean(m_numbers)}')

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRADPORecipeSingleDeviceEval", cfg=cfg)
    recipe = LoRADPORecipeSingleDeviceEval(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
