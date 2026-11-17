
from config import *

from src.tinyllm.model import TinyLLM
from src.tinyllm.tokenizer import Tokenizer
from src.tinyllm.dataset import TextDataset
from src.tinyllm.trainer import Trainer
from src.tinyllm.generation import generate_text
from src.tinyllm.checkpoint import save_checkpoint


def main():
    print(f"Using device: {DEVICE}")

    # Tokenizer
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Dataset
    dataset = TextDataset(
        train_path=TRAIN_BIN_PATH,
        val_path=VAL_BIN_PATH,
        raw_path=RAW_DATA_PATH,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        device=DEVICE
    )

    # Model
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER
    ).to(DEVICE)

    # Generate before training
    print("\n--- Before training ---")
    print(generate_text(
        model, tokenizer, "hello ",
        MAX_NEW_TOKENS, DEVICE
    ))

    # Training
    trainer = Trainer(
        model=model,
        dataset=dataset,
        learning_rate=LEARNING_RATE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        device=DEVICE
    )

    trainer.train(
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE
    )

    # Generate after training
    print("\n--- After training ---")
    print(generate_text(
        model, tokenizer, "hello ",
        MAX_NEW_TOKENS, DEVICE
    ))

    # Save checkpoint
    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        trainer.optimizer,
        {
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER
        },
        tokenizer.vocab_size
    )

    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
