import os
import multiprocessing
# Setting cache directories
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import logging
from src.train_modes.train_mlm import MLM
from src.train_modes.train_aer import AER
from src.train_modes.train_da_bt import DA_BT
from src.train_modes.train_ft import FT
from src.train_modes.evaluation import Evaluate


def train_mlm(args: argparse.ArgumentParser):
    args.logger.info("---Training MLM---")
    args.train_mode = "mlm"
    mlm_model = MLM(args)
    mlm_model.train()
    
    # train tokenizer will happen only once
    # for subsequence training objective, tokenizer will not be trained
    args.train_tokenizer = False

    # set train_from_scratch to False to prevent training from scratch for the next training objective
    args.train_from_scratch = False


def train_aer(args: argparse.ArgumentParser):
    args.logger.info("---Training AER---")
    args.train_mode = "aer"
    dataset_format = args.dataset_format
    # AER format is json
    args.dataset_format = "json"
    aer_model = AER(args)
    aer_model.train()
    args.dataset_format = dataset_format


def train_bt(args: argparse.ArgumentParser):
    args.logger.info("---Training DAE and BT---")
    args.train_mode = "bt"
    #args.dataset_format = "tok"

    # Data collator fails on num_process > 0, so we set num_process to 0
    args.num_process = 0
    dae_bt_model = DA_BT(args=args)
    dae_bt_model.train()

def train_fine_tune(args: argparse.ArgumentParser):
    args.logger.info("---Fine-Tuning on synthetic data---")
    args.train_mode = "ft"
    # we expect json file for synthetic data or fine-tune dataset
    dataset_format = args.dataset_format
    args.dataset_format = "jsonl"
    ft_model = FT(args=args)
    ft_model.train()
    args.dataset_format = dataset_format

def evaluate(args: argparse.ArgumentParser):
    args.logger.info("---Evaluation---")
    args.train_mode = "eval"
    #args.dataset_format = "tok"

    # Data collator fails on num_process > 0, so we set num_process to 0
    args.num_process = 0
    args.num_train_epochs_bt = 1
    evaluation_model = Evaluate(args=args)
    evaluation_model.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CodeRosetta: Automatic Translation"
    )

    parser.add_argument(
        "--log_mode",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Default logging mode",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=16,
        help="Number of processors data loaders",
    )
    parser.add_argument(
        "--tokenizer_num_process",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processors used for tokenizing dataset",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path", type=str, default="dataset", help="path to the dataset folder"
    )


    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch. Optional: 'epoch' for the each epoch or a number for the number of steps.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint, if the training should continue from a checkpoint folder.",
    )
    
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="tok",
        help="file extention of the dataset. C2CUDA dataset files are tok files.",
    )

    parser.add_argument(
        "--num_beam",
        type=int,
        default=1,
        help="Number of beams for search",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of return candidates.",
    )

    parser.add_argument(
        "--keep_in_memory",
        action="store_true",
        default=False,
        help="Load the whole dataset to memory",
    )
    parser.add_argument("--make_dataset_balance", action="store_true", default=True, help="Equals the number of training samples for language A and B")
    parser.add_argument(
        "--langs", nargs="+", default=["cpp", "cuda"], help="src and targe languages"
    )
    parser.add_argument(
        "--tokenizer_batch_size",
        type=int,
        default=10000,
        help="Batch size for tokenizer.map function",
    )

    # Dataset downsampling
    parser.add_argument(
        "--train_downsampling_limit",
        type=int,
        default=0,
        help="Shrinking trainset size for testing purposes. set 0 to disable",
    )
    parser.add_argument(
        "--test_downsampling_limit",
        type=int,
        default=0,
        help="Shrinking testset size for testing purposes",
    )

    # Base architecture. By default using microsoft/unixcoder-base-nine architecture.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="microsoft/unixcoder-base-nine",
        help="base transformer model",
    )
    
    parser.add_argument(
        "--tokenizer_dir", type=str, default="coderosetta_tokenizer", help="Path to save or load tokenizer"
    )
    parser.add_argument(
        "--train_tokenizer", action="store_true", help="Train the tokenizer on the dataset", default=True
    )
    parser.add_argument("--top_k_tokens", type=int, default=3000, help="Top k tokens for random token insertion")
    parser.add_argument(
        "--output_dir", type=str, default="CodeRosetta_output", help="output directory path to save model"
    )
    parser.add_argument("--train_from_scratch", default=True, action="store_true", help="By default the model is trained from scratch")
    parser.add_argument(
        "--quant",
        type=str,
        default="fp16",
        help="specificy the quantization e.g: torch.bfloat16",
    )

    # training arguments
    parser.add_argument("--whole_word_masking_mlm", action="store_true", default=True)

    parser.add_argument(
        "--train_mode", nargs="+", choices=["mlm", "aer", "bt", "ft", "eval"], default=["mlm", "bt", "ft", "eval"]
    )

    parser.add_argument(
        "--only_bt", action="store_true", help="This disables DAE and only enables BT"
    )

    parser.add_argument(
        "--only_dae", action="store_true", help="This disables BT and only enables BT"
    )

    parser.add_argument(
        "--wwm_probability",
        type=float,
        default=0.15,
        help="Whole word masking probability",
    )
    parser.add_argument(
        "--shuffle_batch_order",
        action="store_true",
        default=False,
        help="Shuffling batch order for seq2seq training",
    )
    parser.add_argument(
        "--shuffle_within_batches",
        action="store_true",
        default=True,
        help="Shuffles samples within batches",
    )


    parser.add_argument("--learning_rate_mlm", type=float, default=0.00008)
    parser.add_argument("--learning_rate_aer", type=float, default=0.000005)
    parser.add_argument("--learning_rate_bt", type=float, default=0.00005)
    parser.add_argument("--learning_rate_ft", type=float, default=0.00004)


    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Overwrites the percent_warmup_steps")
    parser.add_argument("--percent_warmup_steps", type=float, default=0.01)
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ],
        default="inverse_sqrt",
    )

    # Chunk_Size equals max_length which by default is 512.
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="length of each sample or Max_length",
    )

    parser.add_argument("--enable_early_stopping", action="store_true", help="Enable early stopping. By default disabled")
    parser.add_argument("--early_stopping_threshold", type=int, default=0, help="How much the specified metric must improve to satisfy early stopping conditions?")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Set threshold to a big number if you don't what early stopping")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_gpu_batch_size", type=int, default=16, help="if max_gpu_batch_size < batch_size then calculates the accumulation steps automatically")
    parser.add_argument("--num_train_epochs_mlm", type=int, default=100, help="If early stopping is active, you can set num_epoch to a big number.")
    parser.add_argument("--num_train_epochs_aer", type=int, default=10, help="If early stopping is active, you can set num_epoch to a big number.")
    parser.add_argument("--num_train_epochs_bt", type=int, default=20, help="If early stopping is active, you can set num_epoch to a big number.")
    parser.add_argument("--num_train_epochs_ft", type=int, default=10, help="If early stopping is active, you can set num_epoch to a big number.")
    #parser.add_argument("--num_train_epochs", type=int, default=15, help="If early stopping is active, you can set num_epoch to a big number.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If maximum step is set, num_train_epochs will be ignored",
    )
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")

    # DAE and BT configureation
    parser.add_argument("--dae_warmup_setps", type=int, default=0, help="During warmup phase, only DAE will be applied, no BT.")
    parser.add_argument("--dae_word_mask", type=float, default=0.15, help="0 means disabled")
    parser.add_argument("--dae_word_dropout", type=float, default=0.20, help="0 means disabled")
    parser.add_argument("--dae_word_replacement", type=float, default=0, help="0 means disabled")
    parser.add_argument("--dae_word_insertion", type=float, default=0.15, help="0 means disabled")
    parser.add_argument("--dae_word_shuffle", type=float, default=1, help="0 means disabled")
    parser.add_argument("--ratio_steps_update", type=str, default=None, help="When DAE ratio should update. Can be set to number of steps or epoch or half_epoch or quarter_epoch. None means disabled.")
    parser.add_argument("--ratio_percent_update", type=float, default=0.05, help="percent of changes for updating ratios of word replacement and masking for DAE")
    parser.add_argument("--ratio_percent_update_dropout", type=float, default=0.025, help="percent of changes for updating ratio of word droppingfor DAE")
    parser.add_argument("--max_corruption_percent", type=float, default=0.6, help="What percent of input sentence can be corrupted at max?")

    args = parser.parse_args()
    args.dataset_path = os.path.abspath(args.dataset_path)

    # Deciding accumulation step size
    if args.batch_size > args.max_gpu_batch_size:
        args.accumulation_steps = args.batch_size // args.max_gpu_batch_size
        args.batch_size = args.max_gpu_batch_size
    else:
        args.accumulation_steps = 1

    args.logger = logging.getLogger()
    logging.basicConfig(level=args.log_mode.upper())
    args.logger.info(f'Max Batch Size: {args.batch_size}, Accumulation Steps: {args.accumulation_steps}')
    args.logger.info(args)
    for train_mode in args.train_mode:
        if train_mode == "mlm":
            train_mlm(args)
        elif train_mode == "aer":
            train_aer(args)
        elif train_mode == "bt":
            train_bt(args)
        elif train_mode == "ft":
            train_fine_tune(args)
        elif train_mode == "eval":
            evaluate(args)