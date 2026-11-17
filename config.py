
from pathlib import Path
import torch

# Project paths
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TOKENIZER_PATH = DATA_DIR / "mini.model"
RAW_DATA_PATH = DATA_DIR / "train_small.txt"

TRAIN_BIN_PATH = PROCESSED_DIR / "train.bin"
VAL_BIN_PATH = PROCESSED_DIR / "val.bin"

CHECKPOINT_PATH = CHECKPOINT_DIR / "tiny_llm_checkpoint.pth"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters
BLOCK_SIZE = 1024
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2

# Training hyperparameters
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-3
MAX_STEPS = 10

# Generation
MAX_NEW_TOKENS = 50
