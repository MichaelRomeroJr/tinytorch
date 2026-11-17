
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import *
from src.tinyllm.model import TinyLLM
from src.tinyllm.tokenizer import Tokenizer
from src.tinyllm.generation import generate_text

# checkpoint = torch.load(
#     CHECKPOINT_PATH,
#     map_location=DEVICE,
#     weights_only=True
# )
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

tokenizer = Tokenizer(TOKENIZER_PATH)

model_config = checkpoint["config"]

model = TinyLLM(
    vocab_size=checkpoint["vocab_size"],
    **model_config
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "hello "

print(generate_text(
    model,
    tokenizer,
    prompt,
    MAX_NEW_TOKENS,
    DEVICE
))
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import *
from src.tinyllm.model import TinyLLM
from src.tinyllm.tokenizer import Tokenizer
from src.tinyllm.generation import generate_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using the trained TinyLLM."
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="hello ",
        help="Text prompt to start generation (default: 'hello ')."
    )

    args = parser.parse_args()

    # checkpoint = torch.load(
    #     CHECKPOINT_PATH,
    #     map_location=DEVICE,
    #     weights_only=True
    # )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    tokenizer = Tokenizer(TOKENIZER_PATH)

    model_config = checkpoint["config"]

    model = TinyLLM(
        vocab_size=checkpoint["vocab_size"],
        **model_config
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Prompt: {args.prompt}\n")

    print(generate_text(
        model,
        tokenizer,
        args.prompt,
        MAX_NEW_TOKENS,
        DEVICE
    ))


if __name__ == "__main__":
    main()
