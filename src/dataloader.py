import os
from typing import Optional

from datasets import disable_progress_bar, load_dataset


def load(
    tokenizer,
    seq_len,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
    progressbar: Optional[bool] = True,
):
    if not progressbar:
        disable_progress_bar()

    train_data_path = os.path.abspath(train_data_path)
    if os.path.isdir(train_data_path):
        pass
    is_eval = False

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = os.path.abspath(eval_data_path)
        is_eval = True

    data = load_dataset(".", data_files=datafiles)

    # =========== user define ===========
    def _process(batch):
        enc_tokenized = tokenizer(
            batch["input_text"],
            max_length=seq_len,
            # padding="max_length",
            truncation=True,
        )
        dec_tokenized = tokenizer(
            batch["labels"],
            max_length=seq_len,
            # padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": enc_tokenized.input_ids,
            "attention_mask": enc_tokenized.attention_mask,
            "labels": dec_tokenized.input_ids,
        }

    data = data.map(
        _process,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )
    # =========== end user define ===========

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    if train_test_split is not None:
        assert 0.0 < train_test_split < 1.0, "train_test_split must be a value between 0 and 1"
        data = data["train"].train_test_split(train_size=train_test_split, shuffle=False)
        is_eval = True

    return (data["train"], data["test"] if is_eval else None)
