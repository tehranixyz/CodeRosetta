from transformers import (
    get_scheduler,
    EncoderDecoderModel,
    GenerationConfig,
    DefaultDataCollator,
)
import evaluate
from accelerate import Accelerator

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from accelerate import DistributedDataParallelKwargs

import pandas as pd

import numpy as np

from tqdm.auto import tqdm

from ..util.utility import (
    EarlyStoppingCallback,
    shift_tokens_right,
)


class FT:
    def __init__(self, args):
        self.args = args
        if "fortran" in self.args.langs:
            from ..dataset.dataset_fortran_cpp import CodeRosettaDataset
        else:
            from ..dataset.dataset_cpp_cuda_ft import CodeRosettaDataset

        # Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.args.quant,
            split_batches=False,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
            gradient_accumulation_steps=self.args.accumulation_steps,
        )

        with self.accelerator.main_process_first():
            self.dataset = CodeRosettaDataset(args=args)

            # Setting the input output directory for the model
            # self.input_directory = args.output_dir + "_daebt_cuda"
            self.input_directory = args.output_dir + "_daebt"
            self.output_dir = args.output_dir + "_ft"
            print(f"Loaded model: {self.input_directory}")

            # Load the model
            self.model = EncoderDecoderModel.from_pretrained(self.input_directory)
        self.accelerator.wait_for_everyone()

        # Creating the data collator object
        # No need to shift labels, as this will happen automatically inside the model
        self.data_collator = DefaultDataCollator(return_tensors="pt")

        # Metric
        self.bleu_metric = evaluate.load("bleu")
        self.codebleu_metric = evaluate.load("k4black/codebleu")
        self.codebleu_weights = (0.25, 0.25, 0.25, 0.25)

    def prepare_data(self):
        train_dataloader = DataLoader(
            self.dataset("train"),
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        eval_dataloader = DataLoader(
            self.dataset("valid"),
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            self.dataset("test"),
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        return train_dataloader, eval_dataloader, test_dataloader

    def train(self):
        self.train_weakly_supervision_using_accelerator()

    def train_weakly_supervision_using_accelerator(self):
        # Early Stopping Callback
        if self.args.enable_early_stopping:
            early_stopping = EarlyStoppingCallback(
                threshold=self.args.early_stopping_threshold,
                patience=self.args.early_stopping_patience,
            )

        # Keeping track of best results
        max_best_score = float("-inf")

        train_dataloader, eval_dataloader, test_dataloader = self.prepare_data()

        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate_ft)

        num_update_steps_per_epoch = len(train_dataloader)
        num_epochs = self.args.num_train_epochs_ft
        num_training_steps = num_epochs * num_update_steps_per_epoch
        if self.args.max_steps > 0:
            num_epochs = (self.args.max_steps // num_update_steps_per_epoch) + 1
        self.args.logger.info(f"Number of epochs: {num_epochs}")
        self.args.logger.info(f"Number of training_steps: {num_training_steps}")

        # Setting number of warmup steps
        if self.args.num_warmup_steps > 0:
            num_warmup_steps = self.args.num_warmup_steps
        elif self.args.percent_warmup_steps > 0:
            num_warmup_steps = int(num_training_steps * self.args.percent_warmup_steps)
        else:
            num_warmup_steps = 0

        self.args.logger.info(f"Number of warmup steps: {num_warmup_steps}")

        lr_scheduler = get_scheduler(
            self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps // self.args.accumulation_steps,
        )

        (
            self.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lr_scheduler,
        )

        progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

        self.args.logger.info("Begining the training loop.")
        for epoch in range(num_epochs):

            # Training
            self.model.train()
            for batch in train_dataloader:
                # Defining decoder start token based on the language of the batch
                src_lan = f'{self.dataset.tokenizer.convert_ids_to_tokens([batch["lang"][0]])[0]}'
                if src_lan == self.args.langs[0]:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[1].upper()}>"
                        )
                    )
                else:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[0].upper()}>"
                        )
                    )

                if (
                    len(
                        set(self.dataset.tokenizer.convert_ids_to_tokens(batch["lang"]))
                    )
                    != 1
                ):
                    raise ValueError(
                        "Batch should only contains samples of one language!"
                    )
                # if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                #     self.model.module.config.decoder_start_token_id = (
                #         decoder_start_token_id
                #     )
                # else:
                #     self.model.config.decoder_start_token_id = decoder_start_token_id

                decoder_input_ids, decoder_attention_mask = shift_tokens_right(
                    input_ids=batch["labels"].to(torch.int64),
                    pad_token_id=self.dataset.tokenizer.pad_token_id,
                    decoder_start_token_id=decoder_start_token_id,
                )
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"].to(torch.int64),
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    # Gradient Clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    progress_bar.update(1)

            # Validation set

            self.model.eval()
            for batch in tqdm(eval_dataloader):
                src_lan = f'{self.dataset.tokenizer.convert_ids_to_tokens([batch["lang"][0]])[0]}'
                if src_lan == self.args.langs[0]:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[1].upper()}>"
                        )
                    )
                else:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[0].upper()}>"
                        )
                    )

                generation_config = GenerationConfig(
                    max_new_tokens=self.args.chunk_size,
                    decoder_start_token_id=decoder_start_token_id,
                    pad_token_id=self.dataset.tokenizer.pad_token_id,
                    bos_token_id=self.dataset.tokenizer.bos_token_id,
                    eos_token_id=self.dataset.tokenizer.eos_token_id,
                )
                with torch.no_grad():
                    generated_tokens = self.accelerator.unwrap_model(
                        self.model
                    ).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        generation_config=generation_config,
                    )
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens,
                    dim=1,
                    pad_index=self.dataset.tokenizer.pad_token_id,
                )
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )

                predictions_gathered = self.accelerator.gather(generated_tokens)
                labels_gathered = self.accelerator.gather(labels)

                decoded_preds, decoded_labels, decoded_labels_no_bracket = (
                    self.postprocess(predictions_gathered, labels_gathered)
                )
                self.bleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                self.codebleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )

            results = self.bleu_metric.compute()
            code_blue_results = self.codebleu_metric.compute(
                lang="cpp", weights=self.codebleu_weights, tokenizer=None
            )
            self.accelerator.print(
                f"[VALID] Epoch {epoch}: BLEU: {round(results['bleu'],6)*100}, CodeBLEU: {round(code_blue_results['codebleu'],6)*100}"
            )

            testset_decoded_predictions = []
            testset_decoded_labels = []
            # Test set
            for batch in tqdm(test_dataloader):
                src_lan = f'{self.dataset.tokenizer.convert_ids_to_tokens([batch["lang"][0]])[0]}'
                if src_lan == self.args.langs[0]:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[1].upper()}>"
                        )
                    )
                else:
                    decoder_start_token_id = (
                        self.dataset.tokenizer.convert_tokens_to_ids(
                            f"<{self.args.langs[0].upper()}>"
                        )
                    )
                # decoder_start_token_id = self.dataset.tokenizer.convert_tokens_to_ids(decoder_start_token)
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.chunk_size,
                    decoder_start_token_id=decoder_start_token_id,
                    pad_token_id=self.dataset.tokenizer.pad_token_id,
                    bos_token_id=self.dataset.tokenizer.bos_token_id,
                    eos_token_id=self.dataset.tokenizer.eos_token_id,
                )
                with torch.no_grad():
                    generated_tokens = self.accelerator.unwrap_model(
                        self.model
                    ).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        generation_config=generation_config,
                    )
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens,
                    dim=1,
                    pad_index=self.dataset.tokenizer.pad_token_id,
                )
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )

                predictions_gathered = self.accelerator.gather(generated_tokens)
                labels_gathered = self.accelerator.gather(labels)

                decoded_preds, decoded_labels, decoded_labels_no_bracket = (
                    self.postprocess(predictions_gathered, labels_gathered)
                )
                self.bleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                self.codebleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                # TODO: save the result per epoch, not per batch. for some reasons, per epoch throws error at the moment.
                # # Add decoded prediction and labels for later to save them to csv
                testset_decoded_predictions.extend(decoded_preds)
                testset_decoded_labels.extend(decoded_labels_no_bracket)

            results = self.bleu_metric.compute()
            code_blue_results = self.codebleu_metric.compute(
                lang="cpp", weights=self.codebleu_weights
            )
            self.accelerator.print(
                f"[TEST] Epoch {epoch}: BLEU: {round(results['bleu'],6)*100}, CodeBLEU: {round(code_blue_results['codebleu'],6)*100}"
            )

            # Early Stopping check
            if self.args.enable_early_stopping:
                if early_stopping.check_early_stopping(results["bleu"]):
                    print(f"----Early Stopping at epoch {epoch}----")
                    self.accelerator.set_breakpoint()

                if self.accelerator.check_breakpoint():
                    break

            # Always the model with the best results will be saved
            if code_blue_results["codebleu"] > max_best_score:
                max_best_score = code_blue_results["codebleu"]

                # Save the model
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    self.output_dir, save_function=self.accelerator.save
                )
                if self.accelerator.is_main_process:
                    self.dataset.tokenizer.save_pretrained(self.args.tokenizer_dir)
                    self.args.logger.info("Found new best result. Saving the model...")

                    try:
                        data = {
                            "Label": testset_decoded_labels,
                            "Prediction": testset_decoded_predictions,
                        }
                        df = pd.DataFrame(data)
                        df.to_csv(
                            f"test_set_prediction_ws_{epoch}_{self.args.langs[1]}.csv",
                            index=False,
                        )
                    except Exception as e:
                        print(e)

    def postprocess(self, predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = self.dataset.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.dataset.tokenizer.pad_token_id)
        decoded_labels = self.dataset.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        decoded_labels_no_bracket = [label[0].strip() for label in decoded_labels]
        return decoded_preds, decoded_labels, decoded_labels_no_bracket
