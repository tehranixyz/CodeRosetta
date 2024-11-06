import csv
import os, re
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk


from collections import Counter

from ..util.utility import interleave_map_style_datasets_batchwise
from ..util.langs_keyword import langs_keywords
from ..train_modes.train_tokenizer import train_tokenizer


def _removing_repo_address(examples):
    clean = []
    for example in examples["text"]:
        clean.append(example.split("|", 1)[1].strip())
    return {"text": clean}


# Aligning labels with token for AER training task
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


def _shifting_labels(examples):
    # entity_groups = [
    #         'O'
    #     'B-VAR',
    #     'I-VAR',
    #     'B-FUNC',
    #     'I-FUNC'
    #     ...
    #     ]
    new_labels = []
    for label in examples["tags"]:
        new_labels.append([3 if l == 2 else l for l in label])
    return {"tags": new_labels}


class CodeRosettaDataset:

    def __init__(self, args) -> None:
        self.args = args

        # if tokenizer_directory exists, load it, otherwise, the tokenizer will be trained
        if os.path.isdir(self.args.tokenizer_dir):
            self.args.logger.info("Loading Tokenizer")
            tokenizer_checkpoint = args.tokenizer_dir
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_checkpoint,
                model_max_length=512,
                trust_remote_code=True,
                add_prefix_space=True if self.args.train_mode == "aer" else False,
            )

        # Checking for pre-tokenized dataset
        if self.args.train_mode == "mlm":
            tokenized_dataset_path = os.path.join(
                self.args.dataset_path,
                f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                "mlm",
            )
        if self.args.train_mode == "aer":
            tokenized_dataset_path = os.path.join(
                self.args.dataset_path,
                f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                "aer",
            )
        # DAE and BT modes use the same dataset
        if self.args.train_mode == "bt":
            tokenized_dataset_path = os.path.join(
                self.args.dataset_path,
                f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                "bt",
            )
        if os.path.isdir(tokenized_dataset_path):
            self.dataset = load_from_disk(tokenized_dataset_path)
            return

        # Reading dataset files
        datasets = self._load_datasets_from_directory()

        # if self does not have tokenizer attribute, it means tokenizer has not been found and loaded, so let's train it.
        if not hasattr(self, "tokenizer"):
            self.args.logger.info(
                "Loading Tokenizer from checkpoint and making it ready for training."
            )
            tokenizer_checkpoint = args.checkpoint
            # Special tokens to represetn the languages. Examples: <CUDA>, <CPP>
            self.langs = [f"<{lang.upper()}>" for lang in self.args.langs]
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_checkpoint,
                model_max_length=512,
                trust_remote_code=True,
                add_prefix_space=True if self.args.train_mode == "aer" else False,
            )
            self.tokenizer.add_tokens(self.langs, special_tokens=True)
            # Training the tokenizer
            self.args.logger.info("Training Tokenizer")
            self.args.logger.info(
                f"Len tokenizer before training: {len(self.tokenizer)}"
            )
            self.tokenizer = train_tokenizer(self.tokenizer, datasets)
            self.args.logger.info(
                f"Len tokenizer after training: {len(self.tokenizer)}"
            )
            self.tokenizer.save_pretrained(self.args.tokenizer_dir)
            # set train_tokenizer to false to prevent other processors to train the tokenizer
            self.args.train_tokenizer = False

        # Filtering CUDA dataset based on the CUDA keywords
        for dataset in datasets:
            if dataset["train"]["lang"][0] == "cuda":
                self.args.logger.info("Filtering CUDA dataset based on keywords")
                len_before_filtering = len(dataset["train"])
                dataset["train"] = dataset["train"].filter(self._cuda_filter)
                self.args.logger.info(
                    f"Filtered {len_before_filtering - len(dataset['train'])} samples."
                )
                self.args.logger.info(
                    f"Len CUDA dataset from BabelTower: {len(dataset['train'])}"
                )

        # Removing the repository from the training samples
        # This is only for tok files and BabelTower data
        if args.dataset_format == "tok":
            self.args.logger.info("Removing Repos from training data")
            for dataset in datasets:
                dataset["train"] = dataset["train"].map(
                    _removing_repo_address,
                    batched=True,
                    num_proc=self.args.tokenizer_num_process,
                    batch_size=self.args.tokenizer_batch_size,
                )

        # Balancing the two datasets
        if args.make_dataset_balance:
            self.args.logger.info("Making Dataset Balance")
            len_smaller_dataset = min([len(dataset["train"]) for dataset in datasets])
            for dataset in datasets:
                if len(dataset["train"]) > len_smaller_dataset:
                    dataset["train"] = (
                        dataset["train"].shuffle().select(range(len_smaller_dataset))
                    )

        # Downsampling per dataset only for test purposes
        if self.args.train_downsampling_limit > 0:
            self.args.logger.info(
                f"Downsampling trainset to {self.args.train_downsampling_limit}"
            )
            for dataset in datasets:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle()
                    .select(range(args.train_downsampling_limit))
                )
        if self.args.test_downsampling_limit > 0:
            self.args.logger.info(
                f"Downsampling test and validation set to {self.args.test_downsampling_limit}"
            )
            for dataset in datasets:
                dataset["test"] = (
                    dataset["test"]
                    .shuffle()
                    .select(range(args.test_downsampling_limit))
                )
                dataset["valid"] = (
                    dataset["valid"]
                    .shuffle()
                    .select(range(args.test_downsampling_limit))
                )

        # concatinate the datasets into one dataset
        if args.train_mode in ["mlm", "aer"]:
            # we don't need to track the language of each dataset
            # just concat the two datasets
            self.args.logger.info("Concatenating the datasets into one dataset.")
            concatenated_dataset = DatasetDict()
            for split in datasets[0].keys():
                concatenated_dataset[split] = concatenate_datasets(
                    [dataset[split] for dataset in datasets]
                )
                concatenated_dataset[split] = concatenated_dataset[split].shuffle()

            if args.train_mode == "mlm":
                self.args.logger.info("Tokenizing for MLM")
                dataset["train"] = concatenated_dataset["train"].map(
                    self._tokenize_for_mlm,
                    batched=True,
                    batch_size=self.args.tokenizer_batch_size,
                    remove_columns=concatenated_dataset["train"].column_names,
                    num_proc=self.args.tokenizer_num_process,
                )

                dataset["valid"] = concatenated_dataset["valid"].map(
                    self._tokenize_for_mlm_test,
                    batched=True,
                    batch_size=self.args.tokenizer_batch_size,
                    remove_columns=concatenated_dataset["train"].column_names,
                    num_proc=self.args.tokenizer_num_process,
                )
                dataset["test"] = concatenated_dataset["test"].map(
                    self._tokenize_for_mlm_test,
                    batched=True,
                    batch_size=self.args.tokenizer_batch_size,
                    remove_columns=concatenated_dataset["train"].column_names,
                    num_proc=self.args.tokenizer_num_process,
                )
                self.args.logger.info("Grouping texts for MLM.")
                self.dataset = dataset.map(
                    self._group_text,
                    batched=True,
                    num_proc=self.args.tokenizer_num_process,
                )
                # saving tokenized dataset, so that next time we don't need to tokenize it
                self.dataset.save_to_disk(
                    os.path.join(
                        self.args.dataset_path,
                        f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                        "mlm",
                    ),
                    max_shard_size="5GB",
                )
            elif args.train_mode == "aer":
                self.args.logger.info("Tokenizing for AER")
                self.dataset = concatenated_dataset.map(
                    self._tokenize_for_aer_and_align_labels,
                    batched=True,
                    batch_size=self.args.tokenizer_batch_size,
                    num_proc=self.args.tokenizer_num_process,
                    remove_columns=concatenated_dataset["train"].column_names,
                )
                self.dataset.save_to_disk(
                    os.path.join(
                        self.args.dataset_path,
                        f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                        "aer",
                    ),
                    max_shard_size="5GB",
                )

        if args.train_mode == "bt":
            self.args.logger.info("Tokenizing for DAE and BT")
            dataset = interleave_map_style_datasets_batchwise(
                [dataset for dataset in datasets], batch_size=self.args.batch_size * 2
            )
            dataset["train"] = dataset["train"].map(
                self._tokenize_for_dae_bt_train,
                batched=True,
                batch_size=self.args.tokenizer_batch_size,
                num_proc=self.args.tokenizer_num_process,
                remove_columns=dataset["train"].column_names,
            )
            dataset["test"] = dataset["test"].map(
                self._tokenize_for_dae_bt_test_valid,
                batched=True,
                batch_size=self.args.tokenizer_batch_size,
                num_proc=self.args.tokenizer_num_process,
                remove_columns=dataset["test"].column_names,
            )
            dataset["valid"] = dataset["valid"].map(
                self._tokenize_for_dae_bt_test_valid,
                batched=True,
                batch_size=self.args.tokenizer_batch_size,
                num_proc=self.args.tokenizer_num_process,
                remove_columns=dataset["valid"].column_names,
            )
            self.dataset = dataset
            if args.train_mode == "bt":
                # saving tokenized dataset, so that next time we don't need to tokenize it
                self.dataset.save_to_disk(
                    os.path.join(
                        self.args.dataset_path,
                        f"tokenized_datasets_{self.args.langs[0]}_{self.args.langs[1]}",
                        "bt",
                    ),
                    max_shard_size="5GB",
                )

            # Calculating token frequency for token insertion task
            if not os.path.isfile(
                os.path.join(
                    self.args.dataset_path,
                    f"dataset_frequent_token_ids_{self.args.langs[0]}_{self.args.langs[1]}.csv",
                )
            ):
                self.args.logger.info(
                    "Calculating the frequency of tokens for each language."
                )
                self.calculate_token_frequency()

    def __call__(self, split=None):
        if split is None:
            return self.dataset
        else:
            return self.dataset[split]

    def _load_datasets_from_directory(self):
        self.args.logger.info("Loading Dataset")
        train_file_name = ".mono.train."
        valid_file_name = ".para.valid."
        test_file_name = ".para.test."
        dataset_file_format = self.args.dataset_format
        dataset_file_type = "text"

        if self.args.train_mode == "aer":
            # self.args.dataset_format (json) + l => jsonl
            dataset_file_type = "json"
            dataset_file_format = self.args.dataset_format + "l"
            train_file_name = train_file_name + "aer."
            valid_file_name = valid_file_name + "aer."
            test_file_name = test_file_name + "aer."

        # Curretly works with the naming convection of lang.mono.split.dataset_format
        datasets = []
        for lang in self.args.langs:
            dataset = load_dataset(
                dataset_file_type,
                data_files={
                    "train": os.path.join(
                        self.args.dataset_path,
                        f"{lang}{train_file_name}{dataset_file_format}",
                    ),
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

    def _group_text(self, examples):
        chunk_size = self.args.chunk_size
        # Concatinate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Dropping the last chunk if it's smaller thant the chunk_size
        if total_length >= chunk_size:
            total_length = (total_length // chunk_size) * chunk_size
        # Split by chunk of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    def _cuda_filter(self, example):
        for keyword in langs_keywords["cuda_keywords"]:
            if "text" in example.keys():
                if keyword in example["text"]:
                    return True
            elif "tokens" in example.keys():
                if keyword in example["tokens"]:
                    return True
            return False

    def _tokenize_for_mlm(self, examples):
        result = self.tokenizer(examples["text"], truncation=True)
        if self.tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    def _tokenize_for_mlm_test(self, examples):
        result = self.tokenizer(
            examples["text"], padding="max_length", max_length=self.args.chunk_size
        )
        if self.tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    def _tokenize_for_aer_and_align_labels(self, examples):
        # tokenization for AER
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.args.chunk_size,
        )
        all_labels = examples["tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(_align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def _tokenize_for_dae_bt_train(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.args.chunk_size,
            padding="max_length",
        )
        tokenized_inputs["lang"] = self.tokenizer.convert_tokens_to_ids(
            examples["lang"]
        )
        return tokenized_inputs

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
        # tokenized_inputs['labels_lang'] = self.tokenizer.convert_tokens_to_ids(examples['labels_lang'])
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

    def comment_remover(self, batch):
        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|#[^\r\n]*(?:\\\r?\n[^\r\n]*)*',
            re.DOTALL | re.MULTILINE,
        )
        result = [re.sub(pattern, replacer, code) for code in batch["content"]]
        return {"text": result}

    def calculate_token_frequency(self):
        lang1_token_frequency = Counter()
        lang2_token_frequency = Counter()

        for example in self.dataset["train"]:
            if example["lang"] == self.tokenizer.convert_tokens_to_ids(
                self.args.langs[0]
            ):
                lang1_token_frequency.update(example["input_ids"])
            else:
                lang2_token_frequency.update(example["input_ids"])

        # We don't want to track the number of occurance of special tokens
        special_token_ids = list(self.tokenizer.added_tokens_decoder.keys())

        lang1_top_words = [
            self.tokenizer.decode(token_id, skip_special_tokens=True)
            for token_id, frequency in lang1_token_frequency.most_common(
                self.args.top_k_tokens
            )
            if token_id not in special_token_ids
        ]
        lang2_top_words = [
            self.tokenizer.decode(token_id, skip_special_tokens=True)
            for token_id, frequency in lang2_token_frequency.most_common(
                self.args.top_k_tokens
            )
            if token_id not in special_token_ids
        ]

        lang1_top_ids = [
            (token_id, frequency)
            for token_id, frequency in lang1_token_frequency.most_common(
                self.args.top_k_tokens
            )
            if token_id not in special_token_ids
        ]
        lang2_top_ids = [
            (token_id, frequency)
            for token_id, frequency in lang2_token_frequency.most_common(
                self.args.top_k_tokens
            )
            if token_id not in special_token_ids
        ]

        # Combine the lists into a list of tuples
        freq_word_data = list(zip(lang1_top_words, lang2_top_words))
        freq_token_id_data = list(zip(lang1_top_ids, lang2_top_ids))

        # Define the CSV file path
        csv_file_top_words = os.path.join(
            self.args.dataset_path,
            f"dataset_frequent_token_{self.args.langs[0]}_{self.args.langs[1]}.csv",
        )
        csv_file_top_token_ids = os.path.join(
            self.args.dataset_path,
            f"dataset_frequent_token_ids_{self.args.langs[0]}_{self.args.langs[1]}.csv",
        )

        # Write the data to the CSV file
        with open(csv_file_top_words, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow(self.args.langs)
            # Write data from the lists
            writer.writerows(freq_word_data)

        # Write the data to the CSV file
        with open(csv_file_top_token_ids, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow(self.args.langs)
            # Write data from the lists
            writer.writerows(freq_token_id_data)
        self.args.logger.info("Frequent tokens and token_ids are saved.")
