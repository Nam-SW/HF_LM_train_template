import os

import hydra
from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

from dataloader import load
from utils import load_model_and_tokenizer, is_seq2seq


@hydra.main(config_path="../config/", config_name="training_cfg", version_base="1.2")
def main(cfg):
    if cfg.ETC.get("wandb_project") is not None:
        os.environ["WANDB_PROJECT"] = cfg.ETC.wandb_project

    model, tokenizer = load_model_and_tokenizer(cfg.MODELCFG)
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATA.dataset)

    args_cls = Seq2SeqTrainingArguments if is_seq2seq(model) else TrainingArguments
    args = args_cls(
        do_train=True,
        do_eval=eval_dataset is not None,
        **cfg.TRAININGARGS,
    )

    if is_seq2seq(model):  # seq2seq
        collator = DataCollatorForSeq2Seq(tokenizer, model, **cfg.DATA.collator)
    else:  # causal lm
        collator = DataCollatorForLanguageModeling(tokenizer, **cfg.DATA.collator)

    trainer_cls = Seq2SeqTrainer if is_seq2seq(model.config) else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(cfg.PATH.output_dir)


if __name__ == "__main__":
    main()
