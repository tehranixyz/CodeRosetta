import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from ..util.langs_keyword import langs_keywords


class CodeRosettaDataset:
    def __init__(self, args) -> None:
        self.args = args
        # tokenizer_checkpoint
        # mlm is the first training objective, so tokenizer will be download from checkpoint if training mode is mlm
        if args.train_mode == "mlm":
            tokenizer_checkpoint = args.checkpoint
            self.langs = [f"<{lang.upper()}>" for lang in self.args.langs]
            self.tokenizer.add_tokens(self.langs, special_tokens=True)
        else:
            tokenizer_checkpoint = args.tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint,
            model_max_length=512,
            trust_remote_code=True,
            add_prefix_space=True if self.args.train_mode == "aer" else False,
        )

        dataset = self._load_and_prepare_dataset_for_fine_tuning()
        self.dataset = dataset.map(
            self._tokenize,
            batched=True,
            batch_size=self.args.tokenizer_batch_size,
            remove_columns=dataset["train"].column_names,
        )

    def __call__(self, split=None):
        if split is None:
            return self.dataset
        else:
            return self.dataset[split]

    def _load_and_prepare_dataset_for_fine_tuning(self):
        # Reading train set
        train_file = os.path.join(
            self.args.dataset_path, "cpp.cuda.train.synthetic.jsonl"
        )
        train_set = load_dataset("json", data_files={"train": train_file})["train"]
        train_set = train_set.rename_column("generated_cuda", "cuda")

        # Reading test and validation sets
        cpp_validation_file = os.path.join(self.args.dataset_path, "cpp.para.valid.tok")
        cpp_test_file = os.path.join(self.args.dataset_path, "cpp.para.test.tok")
        cpp_val_test_set = load_dataset(
            "text", data_files={"valid": cpp_validation_file, "test": cpp_test_file}
        )
        cpp_val_test_set = cpp_val_test_set.rename_column("text", "cpp")

        cuda_validation_file = os.path.join(
            self.args.dataset_path, "cuda.para.valid.tok"
        )
        cuda_test_file = os.path.join(self.args.dataset_path, "cuda.para.test.tok")
        cuda_val_test_set = load_dataset(
            "text", data_files={"valid": cuda_validation_file, "test": cuda_test_file}
        )
        cuda_val_test_set = cuda_val_test_set.rename_column("text", "cuda")

        # Concatinating the test and validation set horizontally
        valid_set = concatenate_datasets(
            [cpp_val_test_set["valid"], cuda_val_test_set["valid"]], axis=1
        )
        test_set = concatenate_datasets(
            [cpp_val_test_set["test"], cuda_val_test_set["test"]], axis=1
        )

        # Creating Dataset Dict
        dataset = DatasetDict()
        dataset["train"], dataset["valid"], dataset["test"] = (
            train_set,
            valid_set,
            test_set,
        )
        return dataset

    def _tokenize(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["cpp"],
            truncation=True,
            max_length=self.args.chunk_size,
            padding=True,
        )

        tokenized_inputs["lang"] = self.tokenizer.convert_tokens_to_ids(["cpp"]) * len(
            examples["cpp"]
        )

        tokenized_inputs["labels"] = self.tokenizer(
            examples["cuda"],
            truncation=True,
            max_length=self.args.chunk_size,
            padding=True,
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

    def _cuda_filter(self, example):
        for keyword in langs_keywords["cuda_keywords_strict"]:
            if "cuda" in example.keys():
                if keyword in example["cuda"]:
                    return True
