from dataclasses import dataclass
from typing import Union


from transformers import (
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
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
from pytorch_lightning.loggers import CSVLogger


@dataclass
class MyArgs:
    devices: int
    num_nodes: int
    # sample 10% of the entire corpus
    dataset_sample: float = 0.1
    test_size: float = 0.1
    model_size: str = "300m"
    text_col: str = "text"
    max_length: int = 16384
    tokenizer: str = "flax-community/papuGaPT2"


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()
    dataset = load_dataset("oscar", "unshuffled_deduplicated_pl", split="train")

    dataset = dataset.train_test_split(test_size=args.dataset_sample)[
        "test"
    ].train_test_split(test_size=args.test_size)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    config = load_config_from_json(f"configs/retnet-{args.model_size}/config.json")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    def tokenize_datset(example):
        input_ids = tokenizer(
            example[args.text_col],
            truncation=True,
            max_length=args.max_length,
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
        batch_size=train_args.per_device_train_batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_args.per_device_eval_batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    model = LanguageModelingModule(config, train_args)

    logger = CSVLogger(save_dir=train_args.output_dir)
    trainer = pl.Trainer(
        default_root_dir=train_args.output_dir,
        logger=logger,
        max_steps=train_args.max_steps,
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        num_nodes=args.num_nodes,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )


class LanguageModelingModule(pl.LightningModule):
    def __init__(self, config, train_args):
        super().__init__()
        self.save_hyperparameters()
        self.train_args = train_args
        self.model = RetNetForCausalLM(config)

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
            lr=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
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
    main()
