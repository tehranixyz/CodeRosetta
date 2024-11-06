from transformers import (
    DataCollatorForLanguageModeling,
    default_data_collator,
    AutoModelForMaskedLM,
    get_scheduler,
    AutoConfig,
    AutoModelForMaskedLM,
)
from accelerate import Accelerator

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import os


from tqdm.auto import tqdm

import math
from ..util.utility import DataCollatorForWholeWordMasking, EarlyStoppingCallback


class MLM:
    def __init__(self, args):
        self.args = args
        if "fortran" in self.args.langs:
            from ..dataset.dataset_fortran_cpp import CodeRosettaDataset
        else:
            from ..dataset.dataset_cpp_cuda import CodeRosettaDataset

        self.accelerator = Accelerator(
            mixed_precision=self.args.quant,
            gradient_accumulation_steps=self.args.accumulation_steps,
        )

        with self.accelerator.main_process_first():
            self.dataset = CodeRosettaDataset(args=args)

            self.output_dir = args.output_dir + "_mlm"

            if args.whole_word_masking_mlm:
                self.data_collator = DataCollatorForWholeWordMasking(
                    tokenizer=self.dataset.tokenizer,
                    wwm_probability=args.wwm_probability,
                )
            else:
                self.data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.dataset.tokenizer,
                    mlm_probability=args.wwm_probability,
                )
            self.args.logger.debug(self.data_collator)
        self.accelerator.wait_for_everyone()

    def insert_random_mask(self, input_batch):
        features = [dict(zip(input_batch, t)) for t in zip(*input_batch.values())]
        masked_inputs = self.data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    def prepare_data(self):
        # Applying insert_random_mask only to test dataset
        if not self.args.whole_word_masking_mlm:
            # remove word_ids as DataCollatorForLanguageModeling does not expect it
            dataset = self.dataset().remove_columns(["word_ids"])
        else:
            dataset = self.dataset()
        eval_dataset = dataset["test"].map(
            self.insert_random_mask,
            batched=True,
            remove_columns=dataset["test"].column_names,
        )

        eval_dataset = eval_dataset.rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )

        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            collate_fn=default_data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        return train_dataloader, eval_dataloader, eval_dataset

    def train(self):
        self.train_mlm_using_accelerator()

    def train_mlm_using_accelerator(self):

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
        min_best_score = float("inf")

        train_dataloader, eval_dataloader, eval_dataset = self.prepare_data()

        if self.args.train_from_scratch:
            self.args.logger.info("Training from scratch")
            config = AutoConfig.from_pretrained(
                self.args.checkpoint,
                vocab_size=len(self.dataset.tokenizer),
                n_ctx=self.args.chunk_size,
                bos_token_id=self.dataset.tokenizer.bos_token_id,
                eos_token_id=self.dataset.tokenizer.eos_token_id,
                hidden_size=1536,
                num_attention_heads=12,
                num_hidden_layers=12,
                intermediate_size=4096,
                trust_remote_code=True,
            )
            model = AutoModelForMaskedLM.from_config(config)
        else:
            self.args.logger.info(f"Loading {self.args.checkpoint}")
            model = AutoModelForMaskedLM.from_pretrained(
                self.args.checkpoint, trust_remote_code=True
            )
            model.resize_token_embeddings(len(self.dataset.tokenizer))

        # number of parameters
        num_parameters = model.num_parameters() / 1_000_000
        print(f"'>>> Model number of parameters: {round(num_parameters)}M'")

        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate_mlm)

        num_update_steps_per_epoch = len(train_dataloader)
        num_epochs = self.args.num_train_epochs_mlm
        if self.args.max_steps > 0:
            num_epochs = (self.args.max_steps // num_update_steps_per_epoch) + 1

        num_training_steps = num_epochs * num_update_steps_per_epoch
        self.args.logger.info(f"Number fo epochs: {num_epochs}")
        self.args.logger.info(f"Number fo training_steps: {num_training_steps}")

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

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
            )
        )

        # We need to keep track of how many total steps we have iterated over
        overall_step = 0
        # We also need to keep track of the stating epoch so files are named properly
        starting_epoch = 0

        # We need to load the checkpoint back in before training here with `load_state`
        # The total number of epochs is adjusted based on where the state is being loaded from,
        # as we assume continuation of the same training script
        if self.args.resume_from_checkpoint:
            if (
                self.args.resume_from_checkpoint is not None
                or self.args.resume_from_checkpoint != ""
            ):
                self.accelerator.print(
                    f"Resumed from checkpoint: {self.args.resume_from_checkpoint}"
                )
                self.accelerator.load_state(self.args.resume_from_checkpoint)
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                self.accelerator.print("Getting the most recent checkpoint.")
                dirs = [f.name for f in os.scandir("checkpoints") if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
                self.accelerator.print("Resume_Steps is ", resume_step)

        progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

        for epoch in range(starting_epoch, num_epochs):

            model.train()

            if (
                self.args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
            ):
                # We need to skip steps until we reach the resumed step
                self.accelerator.print(f"Skipping {resume_step} steps.")
                active_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, resume_step
                )
                overall_step += resume_step
            else:
                # After the first iteration though, we need to go back to the original dataloader
                active_dataloader = train_dataloader

            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)

                    # Gradient Clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if self.accelerator.is_main_process:
                    progress_bar.update(1)
                overall_step += 1

                # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # These are saved to folders named `step_{overall_step}`
                # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # If mixed precision was used, will also save a "scalar.bin" file
                if (isinstance(checkpointing_steps, int)) and (
                    overall_step % checkpointing_steps == 0
                ):
                    output_dir = os.path.join("checkpoints", f"step_{overall_step}")
                    # # We only keep one checkpoint to save space, so let's remove all files and folders
                    # with self.accelerator.main_process_first:
                    #     if self.accelerator.is_main_process():
                    #         if os.path.isdir('checkpoints'):
                    #             shutil.rmtree('checkpoints')
                    self.accelerator.save_state(output_dir)

            # Evaluation
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(
                    self.accelerator.gather(loss.repeat(self.args.batch_size))
                )

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]

            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")

            self.accelerator.print(f"Epoch {epoch}: Perplexity: {perplexity}")

            # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
            # These are saved to folders named `epoch_{epoch}`
            # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
            # If mixed precision was used, will also save a "scalar.bin" file
            if checkpointing_steps == "epoch":
                output_dir = os.path.join("checkpoints", f"epoch_{epoch}")
                # remove checkpoints every 10 epochs:
                # if epoch % 10 == 0:
                #     with self.accelerator.main_process_first:
                #         if self.accelerator.is_main_process():
                #             if os.path.isdir('checkpoints'):
                #                 shutil.rmtree('checkpoints')
                self.accelerator.save_state(output_dir)

            # Early Stopping check
            if self.args.enable_early_stopping:
                if early_stopping.check_early_stopping(perplexity):
                    print(f"----Early Stopping at epoch {epoch}----")
                    self.accelerator.set_breakpoint()

                if self.accelerator.check_breakpoint():
                    break

            # Always the model with the best results will be saved
            if perplexity < min_best_score:
                min_best_score = perplexity

                # Save the model
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    self.output_dir,
                    save_function=self.accelerator.save,
                    is_main_process=self.accelerator.is_main_process,
                )
                if self.accelerator.is_main_process:
                    self.dataset.tokenizer.save_pretrained(self.args.tokenizer_dir)
                    self.args.logger.info("Found new best result. Saving the model...")
