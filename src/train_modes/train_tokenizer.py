from datasets import concatenate_datasets


def get_training_corpus(train_set):
    for start_idx in range(0, len(train_set), 1000):
        samples = train_set[start_idx : start_idx + 1000]
        yield samples["text"]


def train_tokenizer(tokenizer, datasets):
    trainset = concatenate_datasets(
        [dataset["train"] for dataset in datasets]
    ).shuffle()
    # remove non text columns
    trainset = trainset.remove_columns(
        [col for col in trainset.column_names if col != "text"]
    )
    training_corpus = get_training_corpus(train_set=trainset)
    tokenizer = tokenizer.train_new_from_iterator(training_corpus, len(tokenizer))
    return tokenizer
