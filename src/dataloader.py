import os
from typing import Optional

from datasets import disable_progress_bar, load_dataset


def load(
    tokenizer,
    seq_len: int = 512,
    path: Optional[str] = None,
    token: Optional[str] = None,
    train_data_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    length_filtering: bool = True,
    shuffle_seed: Optional[int] = None,
    progressbar: Optional[bool] = True,
):
    if not progressbar:
        disable_progress_bar()

    is_eval = False
    if path is not None:
        data = load_dataset(path, token=token)

    else:
        train_data_path = os.path.abspath(train_data_path)
        if os.path.isdir(train_data_path):
            pass

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
        tokenized = tokenizer(batch["input_text"], text_target=batch["labels"])

        return tokenized

    # Write your own mapping, shuffling, and splitting sequences.

    if train_test_split is not None:
        assert 0.0 < train_test_split < 1.0, "train_test_split must be a value between 0 and 1"
        data = data["train"].train_test_split(train_size=train_test_split, shuffle=False)
        is_eval = True

    data = data.map(
        _process,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    if length_filtering:
        data = data.filter(lambda x: len(x["input_ids"]) <= seq_len)

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    # =========== end user define ===========

    return (data["train"], data["test"] if is_eval else None)
