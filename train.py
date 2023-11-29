from dataclasses import dataclass
import sys

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset
from transformers import HerbertTokenizer
from retnet.modeling_retnet import RetNetForCausalLM
from retnet.configuration_retnet import load_config_from_json


@dataclass
class MyArgs:
    model_size: str = '300m'
    text_col: str = 'text'
    max_length: int = 16384
    tokenizer: str = 'flax-community/papuGaPT2'


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()

    dataset = load_dataset('oscar', 'unshuffled_deduplicated_pl', split="train")
    # sample 10% of the entire corpus
    dataset = dataset.train_test_split(test_size=0.1)["test"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    # eval_dataset = load_dataset('oscar', 'unshuffled_deduplicated_pl', split="validation")

    config = load_config_from_json(f"configs/retnet-{args.model_size}/config.json")
    model = RetNetForCausalLM(config)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    def tokenize_datset(example):
        input_ids = tokenizer(example[args.text_col],
                              truncation=True,
                              max_length=args.max_length,
                              return_tensors='pt').input_ids[0]
        return {'input_ids': input_ids}

    train_dataset = train_dataset.map(tokenize_datset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_datset, remove_columns=eval_dataset.column_names)

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if train_args.do_train:
        trainer.train()
        trainer.save_model()
    if train_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
