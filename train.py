from dataclasses import dataclass
import json
import os
from typing import Union
import time

import typer

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from retnet.modeling_retnet import RetNetForCausalLM
from retnet.configuration_retnet import load_config_from_json
import pytorch_lightning as pl
from torch import Tensor
import torch
import transformers
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


DEFUALT_MODEL = "300m"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TOKENIZER = "flax-community/papuGaPT2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 6e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_STEPS = 20000


def main(
    devices: int = typer.Option(...),
    num_nodes: int = typer.Option(...),
    output_dir: str = typer.Option(...),
    accelerator: str = typer.Option("gpu"),
    dataset_sample: float = typer.Option(0.1),
    test_size: float = typer.Option(0.1),
    check_val_every_n_epochs: int = typer.Option(10),
    model_size: str = typer.Option(DEFUALT_MODEL),
    text_col: str = typer.Option("text"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH),
    tokenizer: str = typer.Option(DEFAULT_TOKENIZER),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE),
    learning_rate: float = typer.Option(DEFAULT_LR),
    weight_decay: float = typer.Option(DEFAULT_WEIGHT_DECAY),
    max_steps: int = typer.Option(DEFAULT_MAX_STEPS),
):
    dataset = load_dataset("oscar", "unshuffled_deduplicated_pl", split="train")

    dataset = dataset.train_test_split(test_size=dataset_sample)[
        "test"
    ].train_test_split(test_size=test_size)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    config = load_config_from_json(f"configs/retnet-{model_size}/config.json")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    def tokenize_datset(example):
        input_ids = tokenizer(
            example[text_col],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"input_ids": input_ids}

    train_dataset = train_dataset.map(
        tokenize_datset, remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        tokenize_datset, remove_columns=eval_dataset.column_names
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    model = LanguageModelingModule(
        config,
        train_args=dict(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    loggers = [
        CSVLogger(save_dir=output_dir),
        TensorBoardLogger(save_dir=output_dir, default_hp_metric=False),
    ]

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        logger=loggers,
        check_val_every_n_epoch=check_val_every_n_epochs,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy="ddp",
        num_nodes=num_nodes,
    )

    start = time.time()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
    end = time.time()

    results = {
        "elapsed_seconds": end - start,
    }

    with open(os.path.join(trainer.log_dir, "results.json"), "w") as file:
        json.dump(results, file, indent="\t")


class LanguageModelingModule(pl.LightningModule):
    def __init__(self, retnet_config, train_args):
        super().__init__()
        self.train_args = train_args
        self.model = RetNetForCausalLM(retnet_config)

    def forward(self, batch) -> tuple:
        output = self.model(**batch)
        return output.loss, output.logits

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, _ = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, _ = self.forward(batch)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_args["learning_rate"],
            weight_decay=self.train_args["weight_decay"],
        )
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(
        self, num_training_steps: int, num_warmup_steps: Union[int, float]
    ) -> tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps


if __name__ == "__main__":
    typer.run(main)
