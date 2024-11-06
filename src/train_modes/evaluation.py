from transformers import (
    EncoderDecoderModel,
    GenerationConfig,
    DefaultDataCollator,
)
import evaluate
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
from accelerate import DistributedDataParallelKwargs
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os


class Evaluate:
    def __init__(self, args):
        self.args = args
        if "cuda" in self.args.langs:
            self.args.logger.info("Loading dataset from dataset_eval")
            from ..dataset.dataset_cpp_cuda_eval import CodeRosettaDataset
        elif "fortran" in self.args.langs:
            self.args.logger.info("Loading dataset from dataset_fortran_cpp")
            from ..dataset.dataset_fortran_cpp import CodeRosettaDataset

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
            self.args.logger.info(self.dataset("test"))

            # if fine-tuned verion exists load it, load DAE-BT model
            self.input_directory = (
                args.output_dir + "_ft"
                if os.path.isdir(args.output_dir + "_ft")
                else args.output_dir + "_daebt"
            )
            self.accelerator.print(f"Input Directory: {self.input_directory}")

            # Load the model
            self.model = EncoderDecoderModel.from_pretrained(self.input_directory)
        self.accelerator.wait_for_everyone()

        # create data collator object
        self.valid_data_collator = DefaultDataCollator(return_tensors="pt")

        # Bleu Metric
        self.bleu_metric = evaluate.load("bleu")
        self.codebleu_metric = evaluate.load("k4black/codebleu")

        # Weights for CodeBLEU
        self.weights = (0.25, 0.25, 0.25, 0.25)

    def prepare_data(self):
        test_dataloader = DataLoader(
            self.dataset("test"),
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=self.valid_data_collator,
            num_workers=self.args.num_process,
            pin_memory=True,
        )

        return test_dataloader

    def evaluate(self):
        self.evaluate_with_accelerator()

    def evaluate_with_accelerator(self):
        test_dataloader = self.prepare_data()
        (
            self.model,
            test_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            test_dataloader,
        )
        max_new_tokens = self.args.chunk_size
        num_beam = self.args.num_beam
        num_return_sequences = self.args.num_return_sequences
        self.accelerator.print(f"Max new tokens: {max_new_tokens}")
        self.accelerator.print(f"Number of beams: {num_beam}")

        self.model.eval()

        testset_decoded_predictions, testset_decoded_labels = [], []
        for batch in tqdm(test_dataloader):
            src_lan = (
                f'{self.dataset.tokenizer.convert_ids_to_tokens([batch["lang"][0]])[0]}'
            )
            if src_lan == self.args.langs[0]:
                decoder_start_token_id = self.dataset.tokenizer.convert_tokens_to_ids(
                    f"<{self.args.langs[1].upper()}>"
                )
            else:
                decoder_start_token_id = self.dataset.tokenizer.convert_tokens_to_ids(
                    f"<{self.args.langs[0].upper()}>"
                )
            generation_config = GenerationConfig(
                max_new_tokens=self.args.chunk_size,
                decoder_start_token_id=decoder_start_token_id,
                pad_token_id=self.dataset.tokenizer.pad_token_id,
                bos_token_id=self.dataset.tokenizer.bos_token_id,
                eos_token_id=self.dataset.tokenizer.eos_token_id,
                num_beams=num_beam,
                num_return_sequences=num_return_sequences,
            )
            with torch.no_grad():
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
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

            decoded_preds, decoded_labels, decoded_labels_no_bracket = self.postprocess(
                predictions_gathered, labels_gathered
            )
            self.bleu_metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels * num_return_sequences,
            )
            self.codebleu_metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels * num_return_sequences,
            )
            testset_decoded_predictions.extend(decoded_preds)
            testset_decoded_labels.extend(
                np.repeat(decoded_labels_no_bracket, num_return_sequences).tolist()
            )

        results = self.bleu_metric.compute()
        code_blue_results = self.codebleu_metric.compute(
            lang="cpp", weights=self.weights, tokenizer=None
        )
        self.accelerator.print(
            f"[TEST]: BLEU: {results['bleu']}, CodeBLEU: {code_blue_results['codebleu']}"
        )

        if self.accelerator.is_main_process:
            try:
                data = {
                    "Label": testset_decoded_labels,
                    "Prediction": testset_decoded_predictions,
                }
                df = pd.DataFrame(data)
                df.to_csv(
                    f"test_set_prediction_{self.args.langs[0]}2{self.args.langs[1]}_beam{num_beam}_seq{num_return_sequences}.csv",
                    index=False,
                    sep="|",
                )
            except Exception as e:
                self.accelerator.print(e)

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
