from transformers import (
    get_scheduler,
    EncoderDecoderModel,
    GenerationConfig,
    DefaultDataCollator,
)
import evaluate
from accelerate import Accelerator
from ..collator.dae_bt_data_collator import DataCollatorForUnsupervisedTranslation

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

from accelerate import DistributedDataParallelKwargs

import pandas as pd

import numpy as np

from tqdm.auto import tqdm

import os
import shutil
from ..util.utility import (
    EarlyStoppingCallback,
    shift_tokens_right,
)


class DA_BT:
    def __init__(self, args):
        self.args = args
        if "fortran" in self.args.langs:
            from ..dataset.dataset_fortran_cpp import CodeRosettaDataset
        else:
            from ..dataset.dataset_cpp_cuda import CodeRosettaDataset

        # Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.args.quant,
            split_batches=False,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(seconds=7200)),
            ],
            gradient_accumulation_steps=self.args.accumulation_steps,
        )

        with self.accelerator.main_process_first():
            self.dataset = CodeRosettaDataset(args=args)

            self.args.logger.debug(
                f"unique lengths in test set: {set([len(self.dataset('test')['input_ids'][i]) for i in range(len(self.dataset('test')))])}"
            )

            # if AER directory exists use it, otherwise use MLM directory
            self.input_directory = (
                args.output_dir + "_aer"
                if os.path.isdir(args.output_dir + "_aer")
                else args.output_dir + "_mlm"
            )
            self.output_dir = args.output_dir + "_daebt"
            self.args.logger.info(f"Loading model from {self.input_directory }")

            # create the model
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.input_directory, self.input_directory
            )
            # self.model.encoder.resize_token_embeddings(len(self.dataset.tokenizer))
            # self.model.decoder.resize_token_embeddings(len(self.dataset.tokenizer))
            # self.model.config.decoder_start_token_id = (
            #     self.dataset.tokenizer.bos_token_id
            # )
            self.model.config.add_cross_attention = True
            self.args.logger.info("Cross Attention is enabled.")
            self.model.config.pad_token_id = self.dataset.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.dataset.tokenizer.eos_token_id
            # number of parameters
            num_parameters = self.model.num_parameters() / 1_000_000
            print(f"'>>> Model number of parameters: {round(num_parameters)}M'")
        self.accelerator.wait_for_everyone()

        # create data collator object
        # Calcualte number of steps per epoch
        if hasattr(self.args.ratio_steps_update, "isdigit"):
            if "epoch" in self.args.ratio_steps_update:
                num_steps_per_epoch = (
                    len(self.dataset("train")) // self.args.batch_size
                ) // self.accelerator.num_processes
                if self.args.ratio_steps_update == "half_epoch":
                    self.args.ratio_steps_update = num_steps_per_epoch // 2
                elif self.args.ratio_steps_update == "quarter_epoch":
                    self.args.ratio_steps_update = num_steps_per_epoch // 4
                else:
                    self.args.ratio_steps_update = num_steps_per_epoch
            elif self.args.ratio_steps_update.isdigit():
                self.args.ratio_steps_update = int(self.args.ratio_steps_update)
            else:
                raise ValueError(
                    f"Argument `ratio_steps_update` must be either a number or `epoch`. `{self.args.ratio_steps_update}` passed."
                )
            self.args.logger.info(f"Ratio_steps_update: {self.args.ratio_steps_update}")

        self.data_collator = DataCollatorForUnsupervisedTranslation(
            args=args,
            accelerator=self.accelerator,
            model=self.model,
            tokenizer=self.dataset.tokenizer,
            langs=self.args.langs,
            word_mask=self.args.dae_word_mask,
            word_dropout=self.args.dae_word_dropout,
            word_replacement_factor=self.args.dae_word_replacement,
            word_insertion_factor=self.args.dae_word_insertion,
            word_shuffle=self.args.dae_word_shuffle,
            max_length=self.args.chunk_size,
            ratio_steps_update=self.args.ratio_steps_update,
            ratio_percent_update=self.args.ratio_percent_update,
            ratio_percent_update_dropout=self.args.ratio_percent_update_dropout,
            max_corruption_percent=self.args.max_corruption_percent,
            num_warmup_setps=self.args.dae_warmup_setps,
            only_dae=self.args.only_dae,
            only_bt=self.args.only_bt,
        )

        self.valid_data_collator = DefaultDataCollator(return_tensors="pt")

        # metric
        self.metric = evaluate.load("bleu")
        self.codebleu_metric = evaluate.load("k4black/codebleu")
        self.weights = (0.25, 0.25, 0.25, 0.25)

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
            collate_fn=self.valid_data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            self.dataset("test"),
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=self.valid_data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        return train_dataloader, eval_dataloader, test_dataloader

    def train(self):
        self.train_dae_bt_using_accelerator()

    def train_dae_bt_using_accelerator(self):

        # Whether we are checkpointing at every epoch or after certain number of steps
        if hasattr(self.args.checkpointing_steps, "isdigit"):
            if self.args.checkpointing_steps == "epoch":
                self.accelerator.print("Considering epochs for checkpointing")
                checkpointing_steps = self.args.checkpointing_steps
            elif self.args.checkpointing_steps.isdigit():
                self.accelerator.print(
                    f"Considering {self.args.checkpointing_steps} for checkpointing"
                )
                checkpointing_steps = int(self.args.checkpointing_steps)
            else:
                raise ValueError(
                    f"Argument `checkpointing_steps` must be either a number or `epoch`. `{self.args.checkpointing_steps}` passed."
                )
        else:
            self.accelerator.print("Checkpointing is disabled.")
            checkpointing_steps = None

        # Early Stopping Callback
        if self.args.enable_early_stopping:
            early_stopping = EarlyStoppingCallback(
                threshold=self.args.early_stopping_threshold,
                patience=self.args.early_stopping_patience,
            )

        # Keeping track of best results
        max_best_score = float("-inf")

        train_dataloader, eval_dataloader, test_dataloader = self.prepare_data()

        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate_bt)

        num_update_steps_per_epoch = len(train_dataloader)
        num_epochs = self.args.num_train_epochs_bt
        num_training_steps = num_epochs * num_update_steps_per_epoch
        if self.args.max_steps > 0:
            num_epochs = (self.args.max_steps // num_update_steps_per_epoch) + 1
        self.args.logger.info(f"Number of epochs: {num_epochs}")
        self.args.logger.info(
            f"Number of steps per epoch: {num_update_steps_per_epoch // self.accelerator.num_processes}"
        )
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

        # We need to keep track of how many total steps we have iterated over
        overall_step = 0
        # We also need to keep track of the stating epoch so files are named properly
        starting_epoch = 0

        # We need to load the checkpoint back in before training here with `load_state`
        # The total number of epochs is adjusted based on where the state is being loaded from,
        # as we assume continuation of the same training script
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "recent":
                self.accelerator.print(
                    f"Resumed from checkpoint: {self.args.resume_from_checkpoint}"
                )
                self.accelerator.load_state(self.args.resume_from_checkpoint)
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                self.accelerator.print("Getting the most recent checkpoint.")
                dirs = [
                    os.path.join("checkpoints", f.name)
                    for f in os.scandir("checkpoints")
                    if f.is_dir()
                ]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]
            self.args.logger.debug(f"training_difference: {training_difference}")

            if "epoch" in training_difference:
                starting_epoch = (
                    int(
                        training_difference.replace(
                            os.path.join("checkpoints", "epoch_"), ""
                        )
                    )
                    + 1
                )
                resume_step = None
            else:
                resume_step = int(
                    training_difference.replace(
                        os.path.join("checkpoints", "step_"), ""
                    )
                )
                self.accelerator.print(f"resume_step: {resume_step}")
                starting_epoch = resume_step // len(train_dataloader)
                self.accelerator.print(f"starting_epoch: {starting_epoch}")
                resume_step -= starting_epoch * len(train_dataloader)
                self.accelerator.print("Resume_Steps is ", resume_step)

        progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

        self.args.logger.info("Begining the training loop.")
        for epoch in range(starting_epoch, num_epochs):

            # Training
            self.model.train()

            if (
                self.args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
            ):
                # We need to skip steps until we reach the resumed step
                self.accelerator.print(f"Skipping {resume_step} steps.")
                progress_bar.update(resume_step)
                active_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, resume_step
                )
                overall_step += resume_step
            else:
                # After the first iteration though, we need to go back to the original dataloader
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                src_lan = f'{self.dataset.tokenizer.convert_ids_to_tokens([batch["lang"][0]])[0]}'
                self.args.logger.debug(f"TRAIN: src_lang: {src_lan}")
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

                self.args.logger.debug(
                    f"TRAIN: decoder_start_token_id: {self.dataset.tokenizer.convert_ids_to_tokens(decoder_start_token_id)}"
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

                # # Old way of training the model by setting the decoder_start_id
                # if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                #     self.model.module.config.decoder_start_token_id = (
                #         decoder_start_token_id
                #     )
                # else:
                #     self.model.config.decoder_start_token_id = decoder_start_token_id

                # outputs = self.model(
                #     input_ids=batch["input_ids"].to(torch.int64),
                #     attention_mask=batch["attention_mask"].to(torch.int64),
                #     labels=batch["labels"].to(torch.int64)
                # )
                self.args.logger.debug(f"shift_tokens_right")
                decoder_input_ids, decoder_attention_mask = shift_tokens_right(
                    input_ids=batch["labels"].to(torch.int64),
                    pad_token_id=self.dataset.tokenizer.pad_token_id,
                    decoder_start_token_id=decoder_start_token_id,
                )
                self.args.logger.debug(f"accelerator.accumulate")
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(torch.int64),
                        attention_mask=batch["attention_mask"].to(torch.int64),
                        labels=batch["labels"].to(torch.int64),
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    self.args.logger.debug(f"outputs")
                    loss = outputs.loss
                    self.args.logger.debug(f"loss")
                    self.accelerator.backward(loss)
                    self.args.logger.debug(f"backward")

                    # Gradient Clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.args.logger.debug(f"clip_grad_norm_")
                    # self.accelerator.clipgrad_norm(self.model.parameters(), 1.0)

                    optimizer.step()
                    self.args.logger.debug(f"optimizer.step()")
                    lr_scheduler.step()
                    self.args.logger.debug(f"lr_scheduler.step()")
                    optimizer.zero_grad()
                    self.args.logger.debug(f"optimizer.zero_grad()")
                if self.accelerator.is_main_process:
                    progress_bar.update(1)
                overall_step += 1

                # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # These are saved to folders named `step_{overall_step}`
                # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # If mixed precision was used, will also save a "scalar.bin" file
                if isinstance(checkpointing_steps, int):
                    output_dir = f"step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        output_dir = os.path.join("checkpoints", output_dir)
                        # deleting older checkpoints
                        if self.accelerator.is_main_process:
                            for dir_name in os.listdir("checkpoints"):
                                dir_path = os.path.join("checkpoints", dir_name)
                                if os.path.isdir(dir_path) and output_dir != dir_path:
                                    shutil.rmtree(dir_path)
                        self.accelerator.save_state(output_dir)

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
                self.metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                self.codebleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )

            results = self.metric.compute()
            code_blue_results = self.codebleu_metric.compute(
                lang="cpp", weights=self.weights, tokenizer=None
            )
            self.accelerator.print(
                f"[VALID] Epoch {epoch}: BLEU: {results['bleu']}, CodeBLEU: {code_blue_results['codebleu']}"
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
                self.metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                self.codebleu_metric.add_batch(
                    predictions=decoded_preds, references=decoded_labels
                )
                # # Add decoded prediction and labels for later to save them to csv
                testset_decoded_predictions.extend(decoded_preds)
                testset_decoded_labels.extend(decoded_labels_no_bracket)

            results = self.metric.compute()
            code_blue_results = self.codebleu_metric.compute(
                lang="cpp", weights=self.weights, tokenizer=None
            )
            self.accelerator.print(
                f"[TEST] Epoch {epoch}: BLEU: {results['bleu']}, CodeBLEU: {code_blue_results['codebleu']}"
            )

            # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
            # These are saved to folders named `epoch_{epoch}`
            # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
            # If mixed precision was used, will also save a "scalar.bin" file
            if checkpointing_steps == "epoch":
                output_dir = os.path.join("checkpoints", f"epoch_{epoch}")
                # remove older checkpoints
                if self.accelerator.is_main_process:
                    for dir_name in os.listdir("checkpoints"):
                        dir_path = os.path.join("checkpoints", dir_name)
                        if os.path.isdir(dir_path) and dir_path != output_dir:
                            shutil.rmtree(dir_path)
                self.accelerator.save_state(output_dir)

            # Early Stopping check
            if self.args.enable_early_stopping:
                if early_stopping.check_early_stopping(results["bleu"]):
                    print(f"----Early Stopping at epoch {epoch}----")
                    self.accelerator.set_breakpoint()

                if self.accelerator.check_breakpoint():
                    break

            # Always the model with the best results will be saved
            if results["bleu"] > max_best_score:
                max_best_score = results["bleu"]

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
                            f"test_set_prediction_{epoch}_{self.args.langs[1]}.csv",
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
