from typing import Any
import os
from transformers import AutoTokenizer
from datasets import load_dataset

from ..util.utility import interleave_map_style_datasets_batchwise


def _align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


class CodeRosettaDataset:

    def __init__(self, args) -> None:
        self.args = args

        # Reading dataset files
        datasets = self._load_datasets_from_directory()

        self.args.logger.info("Loading Tokenizer")
        tokenizer_checkpoint = args.tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint,
            model_max_length=self.args.chunk_size,
            trust_remote_code=True,
            add_prefix_space=True if self.args.train_mode == "aer" else False,
        )

        dataset = interleave_map_style_datasets_batchwise(
            [dataset for dataset in datasets], batch_size=self.args.batch_size
        )
        dataset["test"] = dataset["test"].map(
            self._tokenize_for_dae_bt_test_valid,
            batched=True,
            batch_size=self.args.tokenizer_batch_size,
            remove_columns=dataset["test"].column_names,
            num_proc=self.args.tokenizer_num_process,
        )
        dataset["valid"] = dataset["valid"].map(
            self._tokenize_for_dae_bt_test_valid,
            batched=True,
            batch_size=self.args.tokenizer_batch_size,
            remove_columns=dataset["valid"].column_names,
            num_proc=self.args.tokenizer_num_process,
        )
        self.dataset = dataset

    def __call__(self, split=None):
        if split is None:
            return self.dataset
        else:
            return self.dataset[split]

    def _load_datasets_from_directory(self):
        self.args.logger.info("Loading Dataset")
        valid_file_name = ".para.valid."
        test_file_name = ".para.test."
        dataset_file_format = "tok"
        dataset_file_type = "text"

        # Works with the naming convection of lang.mono.split.dataset_format
        datasets = []
        for lang in self.args.langs:
            dataset = load_dataset(
                dataset_file_type,
                data_files={
                    "valid": os.path.join(
                        self.args.dataset_path,
                        f"{lang}{valid_file_name}{dataset_file_format}",
                    ),
                    "test": os.path.join(
                        self.args.dataset_path,
                        f"{lang}{test_file_name}{dataset_file_format}",
                    ),
                },
                keep_in_memory=self.args.keep_in_memory,
            )
            for split in dataset:
                dataset[split] = dataset[split].add_column(
                    "lang", [lang] * len(dataset[split])
                )
            datasets.append(dataset)

        return datasets

    def _tokenize_for_dae_bt_test_valid(self, examples):
        # Source language
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.args.chunk_size,
            padding="max_length",
        )
        tokenized_inputs["lang"] = self.tokenizer.convert_tokens_to_ids(
            examples["lang"]
        )

        # Target language
        tokenized_inputs["labels"] = self.tokenizer(
            examples["labels"],
            truncation=True,
            max_length=self.args.chunk_size,
            padding="max_length",
        )["input_ids"]
        labels = tokenized_inputs["labels"]
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [
                label if label != self.tokenizer.pad_token_id else -100
                for label in labels_example
            ]
            labels_with_ignore_index.append(labels_example)
        tokenized_inputs["labels"] = labels_with_ignore_index
        return tokenized_inputs
