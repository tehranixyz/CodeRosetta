from typing import (
    List,
    Optional,
)
from datasets.info import DatasetInfo
from datasets import (
    Dataset,
    NamedSplit,
    concatenate_datasets,
    DatasetDict,
)
import numpy as np

import collections
from transformers import default_data_collator
import torch


wwm_probability = 0.15


def interleave_map_style_datasets_batchwise(
    datasets: List["Dataset"],
    split: Optional[NamedSplit] = None,
    batch_size: Optional[int] = 8,
    shuffle_batch_order=True,
    shuffle_within_batches=True,
    **kwargs,
) -> "Dataset":

    concatenated_datasets = DatasetDict()
    for split in datasets[0].keys():
        datasets_split = [dataset[split] for dataset in datasets]

        if split == "train":
            # For train split, we create batches of on language and alternate between languages
            concatenated_datasets[split] = concatenate_datasets(datasets_split)
            # Let's now build the indices to pass to .select()

            # For example we have 3 datasets with the length of [16, 6, 5]
            lengths = [len(dset) for dset in datasets_split]
            # For example [0, 10, 16]
            offsets = np.cumsum([0] + lengths[:-1])
            # Discard any sample that does not fit into a batch
            num_batches = min(lengths) // batch_size

            # For example if batch_size is 2, batch_indices would be [[0,1], [2, 3]]
            batch_indices = np.arange(num_batches * batch_size).reshape(
                num_batches, batch_size
            )
            # Each offset of 0, 10, 16 will be added to [0,1] resulting in [0, 1, 10, 11, 16, 17]
            # then each offset will be added to [2,3] and the end result would be [0, 1, 10, 11, 16, 17, 2, 3, 12, 13, 18, 19]
            # In this way, the first batch contains elements from first dataset, and second batch from the second dataset and so on
            # shape (num_batch, num_datasets, batch_size)
            indices = batch_indices[:, None, :] + offsets.reshape(-1, 1)
            # shape (num_datasets*num_batches, batch_size)
            indices = indices.reshape(-1, batch_size)
            if shuffle_batch_order:
                np.random.shuffle(indices)
            if shuffle_within_batches:
                # transpose, because we want to shuffle within columns
                indices = np.transpose(indices)
                np.random.shuffle(indices)
                # transpose back
                indices = np.transpose(indices)

            indices = indices.flatten().tolist()
            concatenated_datasets[split] = concatenated_datasets[split].select(
                indices, **kwargs
            )
        else:
            # For test or validation splits we set the input_ids of the second dataset as the labels of the first dataset
            concatenated_datasets[split] = datasets_split[0].add_column(
                "labels", datasets_split[1]["text"]
            )
            concatenated_datasets[split] = concatenated_datasets[split].add_column(
                "labels_lang", datasets_split[1]["lang"]
            )

    return concatenated_datasets


class DataCollatorForWholeWordMasking:
    def __init__(self, tokenizer, wwm_probability=0.15):
        self.tokenizer = tokenizer
        self.wwm_probability = wwm_probability

    def whole_word_masking_data_collator(self, features):
        for feature in features:
            word_ids = feature.pop("word_ids")

            # creating a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, self.wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            # labels of all tokens are -100 except those tokens that are masked
            new_labels = [-100] * len(labels)
            # Which words' indexes would be masked? e.g, [0,0,3] => word at index 3 is masked
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                # mapping[word_id], the masked word corresponds to which tokens indexes?
                for idx in mapping[word_id]:
                    # The label of the masked token is the actual label, all other labels are -100
                    new_labels[idx] = labels[idx]
                    # The token's values is masked now
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)

    def __call__(self, features):
        return self.whole_word_masking_data_collator(features)


def whole_word_masking_data_collator(features, tokenizer):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # creating a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        # labels of all tokens are -100 except those tokens that are masked
        new_labels = [-100] * len(labels)
        # Which words' indexes would be masked? e.g, [0,0,3] => word at index 3 is masked
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            # mapping[word_id], the masked word corresponds to which tokens indexes?
            for idx in mapping[word_id]:
                # The label of the masked token is the actual label, all other labels are -100
                new_labels[idx] = labels[idx]
                # The token's values is masked now
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def sampling_from_trainset(train_set_file, num_of_samples=100000):
    with open(train_set_file) as file:
        lines = [line.rstrip() for line in file]

    np.random.shuffle(lines)
    list_b, list_a = lines[:num_of_samples], lines[num_of_samples:]

    with open(f"excluded_{num_of_samples}_samples_{train_set_file}", "w") as f:
        for line in list_a:
            f.write(f"{line}\n")

    with open(f"{num_of_samples}_samples_{train_set_file}", "w") as f:
        for line in list_b:
            f.write(f"{line}\n")


class EarlyStoppingCallback:
    def __init__(self, threshold=0, patience=5):
        self.min_delta = threshold
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float("inf")

    def check_early_stopping(self, eval_loss):
        delta = self.lowest_loss - eval_loss
        if delta >= self.min_delta:
            self.lowest_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    decoder_attention_mask = shifted_input_ids.new_tensor(
        shifted_input_ids != pad_token_id
    )

    return shifted_input_ids, decoder_attention_mask
