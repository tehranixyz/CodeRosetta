from copy import deepcopy

from transformers import default_data_collator, DataCollatorForLanguageModeling
import numpy as np
import torch
from transformers import GenerationConfig
import csv, os
from ast import literal_eval
from collections import defaultdict

from ..util.langs_keyword import langs_keywords


class DataCollatorForUnsupervisedTranslation:

    def __init__(
        self,
        args,
        accelerator,
        model,
        tokenizer,
        langs=["cpp", "cuda"],
        word_mask=0.15,
        word_dropout=0.05,
        word_replacement_factor=0.05,
        word_insertion_factor=0.05,
        word_shuffle=0.1,
        max_length=512,
        ratio_steps_update=None,
        ratio_percent_update=None,
        ratio_percent_update_dropout=None,
        max_corruption_percent=None,
        num_warmup_setps=0,
        dataset_folder="dataset",
        only_dae=False,
        only_bt=False,
    ):
        self.args = args
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.masked_data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        self.langs = langs
        self.mask_index = tokenizer.mask_token_id
        self.pad_index = tokenizer.pad_token_id
        self.eos_index = tokenizer.sep_token_id
        self.sos_index = tokenizer.cls_token_id
        self.word_mask_factor = word_mask
        self.word_dropout_factor = word_dropout
        self.word_replacement_factor = word_replacement_factor
        self.word_insertion_factor = word_insertion_factor
        self.is_word_shuffle = word_shuffle
        self.ratio_steps_update = ratio_steps_update
        self.ratio_percent_update = ratio_percent_update
        self.ratio_percent_update_dropout = ratio_percent_update_dropout
        self.max_corruption_percent = max_corruption_percent
        self.num_warmup_steps = num_warmup_setps

        self.max_length = max_length

        # Structure:
        # {'cpp':{
        #     'token_id',
        #     'token_prob'
        # }}
        self.freq_token_probability = self.read_frequent_ids(
            os.path.join(
                dataset_folder,
                f"dataset_frequent_token_ids_{self.args.langs[0]}_{self.args.langs[1]}.csv",
            )
        )

        # Alternate between the DAE and BT, 0=> do DAE, 1=> do BT
        self.collator_state = 0
        self.only_dae = only_dae
        self.only_bt = only_bt
        if self.only_bt:
            # set data collator state to Back Translation
            self.collator_state = 1
        elif self.only_dae:
            # set data collator state to Denoising Auto Encoding
            self.collator_state = 0

        # step tracker to update DAE ratios
        self.num_steps = 0

        if self.accelerator.is_main_process:
            if self.num_warmup_steps:
                self.args.logger.info(f"DAE for {self.num_warmup_steps} warmup steps.")
            self.args.logger.info(f"---DAE BT Collator initialized---")
            self.args.logger.info(f"word_mask_factor: {self.word_mask_factor}")
            self.args.logger.info(f"word_dropout_factor: {self.word_dropout_factor}")
            self.args.logger.info(
                f"word_replacement_factor: {self.word_replacement_factor}"
            )
            self.args.logger.info(
                f"word_insertion_factor: {self.word_insertion_factor}"
            )

    def update_step_counter(self):
        self.num_steps += 1
        if (
            self.ratio_steps_update is not None
            and self.ratio_steps_update != 0
            and self.num_steps % self.ratio_steps_update == 0
        ):
            self.update_ratios()

    def update_ratios(self):
        if (
            self.word_mask_factor
            + self.word_dropout_factor
            + self.word_replacement_factor
            + self.word_insertion_factor
        ) < self.max_corruption_percent:
            self.word_mask_factor += self.ratio_percent_update
            # self.word_replacement_factor += self.ratio_percent_update
            self.word_insertion_factor += self.ratio_percent_update
            self.word_dropout_factor += self.ratio_percent_update_dropout
            if self.accelerator.is_main_process:
                self.args.logger.info(f"---DAE ratios updated---")
                self.args.logger.info(f"word_mask_factor: {self.word_mask_factor}")
                self.args.logger.info(
                    f"word_dropout_factor: {self.word_dropout_factor}"
                )
                self.args.logger.info(
                    f"word_replacement_factor: {self.word_replacement_factor}"
                )
                self.args.logger.info(
                    f"word_insertion_factor: {self.word_insertion_factor}"
                )

    def get_before_pad(self, x):
        """Obtain length till pad"""
        try:
            return x.index(self.eos_index)
        except:
            return len(x)

    def word_replacement(self, x, l, lang):
        """Randomly replaces some words with words from target language"""
        if self.word_replacement_factor == 0:
            return x

        self.args.logger.debug("Replacing Words")
        # The number of words that will be replaced
        no_replacements = int(l * self.word_replacement_factor)
        self.args.logger.debug(f"no_replacements Words: {no_replacements}")
        if no_replacements < 1:
            return x

        # indices of the words for replacement
        replacement_seq = np.random.choice(range(1, l), no_replacements, replace=False)
        self.args.logger.debug(f"replacement_seq indices: {no_replacements}")

        # Random words to be inserted
        random_words = np.random.choice(
            self.freq_token_probability[lang]["token_id"],
            size=no_replacements,
            replace=False,
            p=self.freq_token_probability[lang]["token_prob"],
        )
        # self.args.logger.debug(f"Selected Random Tokens: {random_words}")
        x2 = np.array(deepcopy(x))
        np.put(x2, replacement_seq, random_words)

        self.args.logger.debug(
            f"Tokens are replaced with: {self.tokenizer.decode(random_words)}"
        )
        return x2.tolist()

    def word_insertion(self, x, l, lang):
        """Randomly inserts some words from target language"""
        if self.word_insertion_factor == 0:
            return x

        self.args.logger.debug("Inserting Words")
        # The number of words that will be inserted
        no_inserts = int(l * self.word_insertion_factor)
        self.args.logger.debug(f"no_inserts Words: {no_inserts}")
        # if no_inserts + l >= len(x):
        #     no_inserts = len(x) - l
        if no_inserts < 1:
            return x

        # indices of the words for insertion
        insertion_seq = np.random.choice(range(1, l - 1), no_inserts, replace=False)
        self.args.logger.debug(f"insertion_seq indices: {no_inserts}")

        # Random words to be inserted
        random_words = np.random.choice(
            self.freq_token_probability[lang]["token_id"],
            size=no_inserts,
            replace=False,
            p=self.freq_token_probability[lang]["token_prob"],
        )
        # self.args.logger.debug(f"Selected Random Tokens: {random_words}")
        x2 = np.array(deepcopy(x))
        x2 = np.insert(x2, insertion_seq, random_words)

        self.args.logger.debug(
            f"Inserted Tokens: {self.tokenizer.decode(random_words)}"
        )

        # new lenght after inserting tokens
        new_length = self.get_before_pad(x2.tolist()) + 1

        if new_length > len(x):
            x2 = x2[: len(x)]
            x2[len(x) - 1] = self.eos_index
        elif new_length < len(x):
            # number of pad tokens to be added after new lenght
            num_pad = len(x) - new_length
            x2 = np.concatenate([x2[:new_length], [self.pad_index] * num_pad], 0)
        else:
            x2[len(x) - 1] = self.eos_index
        return x2.tolist()

    def word_mask(self, x, l):
        """Randomly mask input words"""
        # define droppable word indices
        if self.word_mask_factor == 0:
            return x

        self.args.logger.debug("Masking words")
        no_mask = int(l * self.word_mask_factor)
        if no_mask < 1:
            return x
        # Skip the first token <s> and the last token </s>
        mask_seq = np.random.randint(1, l, no_mask)

        x2 = deepcopy(x)
        for i in mask_seq:
            x2[i] = self.mask_index
        return x2

    def word_dropout(self, x, l, lang):
        """Randomly drop input words with more focuse on language reserved keywords"""

        # Ajust it depending on your needs.
        reserved_keywords_probability_multiplier = 2

        if self.word_dropout_factor == 0:
            return x
        special_token_ids = list(self.tokenizer.added_tokens_decoder.keys())
        # language reserved keywords should have higher probability to be droped
        lang_keywords_ids = self.tokenizer(
            " ".join(langs_keywords[f"{lang}_keywords"])
        )["input_ids"]
        # Removing special tokens from tokenized lang_keywords_ids
        lang_keywords_ids = [
            id for id in lang_keywords_ids if id not in special_token_ids
        ]
        # probability of every token is 1 by default
        input_ids_probabilty = np.ones(l - 1)
        # probability of reserved tokens is 2x
        input_ids_probabilty[np.in1d(x[1:l], lang_keywords_ids)] = (
            input_ids_probabilty[np.in1d(x[1:l], lang_keywords_ids)]
            * reserved_keywords_probability_multiplier
        )
        # convert 1, 1, 1, 2, 1, 2 etc in input_ids_probabilty to probability distribution
        total = sum(input_ids_probabilty)
        input_ids_probabilty = input_ids_probabilty / total

        self.args.logger.debug("Droping Words")
        # define droppable word indices
        no_drops = int(l * self.word_dropout_factor)
        if no_drops < 1:
            return x
        drop_seq = np.random.choice(
            range(1, l), no_drops, replace=False, p=input_ids_probabilty
        )

        x2 = deepcopy(x)
        self.args.logger.debug(
            f"Dropping tokens:\n{self.tokenizer.decode(np.array(x2)[drop_seq])}"
        )
        x2 = np.delete(x2, drop_seq, 0)
        x2 = np.concatenate([x2, [self.pad_index] * no_drops], 0)
        x2 = x2.tolist()

        return x2

    def word_shuffle(self, x, l):
        """Randomly shuffle input words."""
        if self.is_word_shuffle == 0 or l < 10:
            return x

        self.args.logger.debug("Shuffling words.")
        self.args.logger.debug(f"Lenght is: {l}")
        # Choose a subsequence to shuffle
        shuffle_seq = np.random.randint(
            1, l, 2
        )  # leave start token, get start and end of shuffl_seq
        while shuffle_seq[0] == shuffle_seq[1]:
            shuffle_seq = np.random.randint(1, l, 2)
        shuffle_seq.sort()

        if shuffle_seq[1] - shuffle_seq[0] < 2:
            return x

        x2 = deepcopy(x)
        x2 = np.concatenate(
            [
                x[: shuffle_seq[0]],
                np.random.permutation(x[shuffle_seq[0] : shuffle_seq[1]]),
                x[shuffle_seq[1] :],
            ],
            axis=0,
        )
        return x2.tolist()

    def add_noise(self, features):
        """
        Add noise to the encoder input.
        """

        self.args.logger.debug("Adding Noise.")
        for feature in features:
            # Define src and trg language
            lang_1_id = self.tokenizer.convert_tokens_to_ids(self.args.langs[0])
            lang_2_id = self.tokenizer.convert_tokens_to_ids(self.args.langs[1])
            src_lang = feature["lang"]
            trg_lang = lang_2_id if src_lang == lang_1_id else lang_1_id

            # Creating label
            feature["labels"] = feature["input_ids"].copy()
            feature["labels"] = np.array(feature["labels"])
            self.args.logger.debug("---Denoising input and output----")
            feature["labels"][feature["labels"] == self.tokenizer.pad_token_id] = -100

            length = self.get_before_pad(feature["input_ids"])
            # word_dropout randomly drops words for src language
            feature["input_ids"] = self.word_dropout(
                feature["input_ids"],
                l=length,
                lang=self.tokenizer.convert_ids_to_tokens(src_lang),
            )
            length = self.get_before_pad(feature["input_ids"])
            # word_replacement inserts random token from target language
            feature["input_ids"] = self.word_replacement(
                x=feature["input_ids"],
                l=length,
                lang=self.tokenizer.convert_ids_to_tokens(trg_lang),
            )
            feature["input_ids"] = self.word_insertion(
                x=feature["input_ids"],
                l=length,
                lang=self.tokenizer.convert_ids_to_tokens(trg_lang),
            )
            length = self.get_before_pad(feature["input_ids"])
            feature["input_ids"] = self.word_shuffle(feature["input_ids"], length)
            feature["input_ids"] = self.word_mask(feature["input_ids"], length)

            # DAE_BT training loop takes the batch src langauge and select the other language as target langauge
            # So here, after adding noises, we should change the language of the batch.
            # For example, if batch language is CPP, training loop with select CUDA as target language
            # However, after adding noises to the CPP examples, we want the target langauge to be CPP.
            # Because we want to reconstruct the original CPP out of noisy CPP. We don't want to generate CUDA.
            # So we change the language of the batch to CUDA and later in the training loop the other language (CPP) will be selected.
            self.args.logger.debug(
                f"Source lang was {self.tokenizer.convert_ids_to_tokens(feature['lang'])}"
            )
            feature["lang"] = trg_lang
            self.args.logger.debug(
                f"Source lang is now {self.tokenizer.convert_ids_to_tokens(feature['lang'])}"
            )
        return default_data_collator(features)

    def back_translate(self, features):
        self.args.logger.debug("Back Translating")

        src_lang_token = self.tokenizer.convert_ids_to_tokens(features[0]["lang"])
        if src_lang_token == self.args.langs[0]:
            # Src lang is langs[0] so target is langs[1]
            trg_lang_token = self.args.langs[1]
        else:
            # src lang is not langs[0], therefore langs[0] is target lang
            trg_lang_token = self.args.langs[0]

        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(
            f"<{trg_lang_token.upper()}>"
        )
        self.args.logger.debug("Creating input for Back Translating")
        langs = torch.IntTensor(
            [self.tokenizer.convert_tokens_to_ids(trg_lang_token)] * len(features)
        )
        input_ids = torch.stack(
            [torch.IntTensor(example["input_ids"]) for example in features]
        )
        attention_mask = torch.stack(
            [torch.IntTensor(example["attention_mask"]) for example in features]
        )
        generation_config = GenerationConfig(
            max_new_tokens=self.max_length,
            decoder_start_token_id=decoder_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.accelerator.wait_for_everyone()
        self.model.eval()
        with torch.no_grad():
            self.args.logger.debug("Creating Noisy Translations")
            generated_tokens = (
                self.accelerator.unwrap_model(self.model)
                .generate(
                    input_ids=input_ids.to(self.accelerator.device),
                    attention_mask=attention_mask.to(self.accelerator.device),
                    generation_config=generation_config,
                )
                .to("cpu")
            )
            self.args.logger.debug("Noisy Translations Generated")
        self.args.logger.debug("Tokenizing and creating a batch")
        noisy_input = [self.tokenizer.decode(ids[1:]) for ids in generated_tokens]
        all_noisy_input = "\n=========".join(
            [self.tokenizer.decode(ids) for ids in generated_tokens]
        )

        self.args.logger.debug(f"---noisy Generated inputs----:\n{all_noisy_input}")
        self.args.logger.debug(f"BT TRG lang: {trg_lang_token}")
        self.args.logger.debug(f"BT SRC lang: {src_lang_token}")
        batch = self.tokenizer(
            noisy_input,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = input_ids.clone().detach()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        batch["lang"] = langs
        self.args.logger.debug("Setting the model back to train")
        self.accelerator.wait_for_everyone()
        self.model.train()
        self.args.logger.debug("Back translation batch creation completed")
        self.args.logger.debug("---BT generated Batch Language---")
        self.args.logger.debug(self.tokenizer.convert_ids_to_tokens(batch["lang"]))

        return batch

    def read_frequent_ids(self, file_path: str) -> dict:
        """Reads CSV file and returns a dictionary"""
        with open(file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="|")
            tokens_frequency_raw = [row for row in csv_reader]
        langs = self.args.langs
        tokens_probability = {lang: defaultdict(list) for lang in langs}
        for i in range(1, len(tokens_frequency_raw)):
            for j in range(len(langs)):
                token_details = literal_eval(tokens_frequency_raw[i][j])
                token_id = token_details[0]
                token_freq = token_details[1]
                tokens_probability[langs[j]]["token_id"].append(token_id)
                tokens_probability[langs[j]]["token_prob"].append(token_freq)
        # Converting token freqeunceis to probabilities
        for j in range(len(langs)):
            total_occurance = sum(tokens_probability[langs[j]]["token_prob"])
            tokens_probability[langs[j]]["token_prob"] = [
                freq / total_occurance
                for freq in tokens_probability[langs[j]]["token_prob"]
            ]

        return tokens_probability

    def __call__(self, features):
        self.update_step_counter()
        if self.only_bt:
            self.args.logger.debug("Only BT Collator")
            return self.back_translate(features)
        elif self.only_dae:
            self.args.logger.debug("Only DAE Collator")
            return self.add_noise(features)
        elif self.num_steps < self.num_warmup_steps:
            self.args.logger.debug("DAE Collator")
            return self.add_noise(features)
        elif self.collator_state == 0:
            self.args.logger.debug("DAE Collator")
            self.collator_state = 1
            return self.add_noise(features)
        else:
            self.args.logger.debug("BT Collator")
            self.collator_state = 0
            return self.back_translate(features)
