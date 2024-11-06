from transformers import (
    get_scheduler,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
import numpy as np
import evaluate
from accelerate import Accelerator

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from tqdm.auto import tqdm

# import config as CFG
from ..util.utility import EarlyStoppingCallback
from ..dataset.dataset_cpp_cuda import CodeRosettaDataset


class AER:
    def __init__(self, args) -> None:
        self.accelerator = Accelerator(
            mixed_precision=args.quant,
            gradient_accumulation_steps=args.accumulation_steps,
        )
        with self.accelerator.main_process_first():
            self.dataset = CodeRosettaDataset(args=args)
            self.args = args
            # load the trained MLM model
            self.input_directory = args.output_dir + "_mlm"
            self.output_dir = args.output_dir + "_aer"

            # self.label_names = ["O", "B-VAR", "I-VAR", "B-FUNC", "I-FUNC"]
            self.label_names = [
                "O",
                "B-VAR",  # 1 = Identifer
                "I-VAR",
                "B-FUNC",  # 3 = Function
                "I-FUNC",
                "B-TYPE",  # 5 = Type Identifer
                "I-TYPE",
                "B-PRIM",  # 7 = Primitive Type (int, float, etc)
                "I-PRIM",
                "B-NUM",  # 9 = Number Literal
                "I-NUM",
                "B-POIN",  # 11 = Pointer Expression/Reference
                "I-POIN",
                "B-PDEC",  # 13 = Pointer Declerator
                "I-PDEC",
                "B-CONS",  # 15 = Constant
                "I-CONS",
            ]

            self.data_collator = DataCollatorForTokenClassification(
                self.dataset.tokenizer
            )
            self.metric = evaluate.load("seqeval")
        self.accelerator.wait_for_everyone()

    def prepare_data(self):
        train_dataloader = DataLoader(
            self.dataset("train"),
            shuffle=True,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        eval_dataloader = DataLoader(
            self.dataset("valid"),
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        return train_dataloader, eval_dataloader

    def train(self):
        self.train_aer_using_accelerator()

    def train_aer_using_accelerator(self):

        # Early Stopping Callback
        if self.args.enable_early_stopping:
            early_stopping = EarlyStoppingCallback(
                threshold=self.args.early_stopping_threshold,
                patience=self.args.early_stopping_patience,
            )

        # Keeping track of best results
        max_best_score = float("-inf")

        train_dataloader, eval_dataloader = self.prepare_data()

        id2label = {i: label for i, label in enumerate(self.label_names)}
        label2id = {v: k for k, v in id2label.items()}

        model = AutoModelForTokenClassification.from_pretrained(
            self.input_directory, id2label=id2label, label2id=label2id
        )

        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate_aer)

        num_update_steps_per_epoch = len(train_dataloader)
        num_epochs = self.args.num_train_epochs_aer
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

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
            )
        )

        progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

        for epoch in range(num_epochs):
            model.train()
            for batch in train_dataloader:
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

            # Evaluation
            model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )

                predictions_gathered = self.accelerator.gather(predictions)
                labels_gathered = self.accelerator.gather(labels)

                true_predictions, true_labels = self.postprocess(
                    predictions_gathered, labels_gathered
                )
                self.metric.add_batch(
                    predictions=true_predictions, references=true_labels
                )

            results = self.metric.compute()
            self.accelerator.print(
                f"epoch {epoch}:",
                {
                    key: results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            # Early Stopping check
            if self.args.enable_early_stopping:
                if early_stopping.check_early_stopping(results["overall_f1"]):
                    print(f"----Early Stopping at epoch {epoch}----")
                    self.accelerator.set_breakpoint()

                if self.accelerator.check_breakpoint():
                    break

            # Always the model with the best results will be saved
            if results["overall_f1"] > max_best_score:
                max_best_score = results["overall_f1"]

                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    self.output_dir, save_function=self.accelerator.save
                )
                if self.accelerator.is_main_process:
                    self.dataset.tokenizer.save_pretrained(self.args.tokenizer_dir)
                    self.args.logger.info("Found new best result. Saving the model...")

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
