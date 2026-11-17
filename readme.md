# TinyTorch-llm

This project implements a GPT-style, decoder-only Transformer from scratch using PyTorch, featuring causal self-attention, token and positional embeddings, SentencePiece tokenization, and a custom training pipeline with gradient accumulation and GPU acceleration.

```
Decoder-only Transformer: Transformer blocks with causal masking and autoregressive generation.
Causal self-attention: Head uses a lower-triangular mask (torch.tril) to prevent attending to future tokens.
```

Designed for experimentation and rapid training on small datasets, the model can be configured to support larger datasets, longer training runs, and more capable hardware.

## Features

- **Transformer Architecture:** Custom implementation of multi-head causal self-attention, feed-forward networks, and positional embeddings.
- **Tokenization:** SentencePiece tokenizer for converting text into model-ready tokens.
- **Data Preprocessing:** Converts raw text into memory-mapped binary datasets with a 90/10 training-validation split.
- **Efficient Training:** Gradient accumulation, mixed-precision training (CUDA), and GPU acceleration when available.
- **Text Generation:** Autoregressive text generation using trained model checkpoints.
- **Configurable:** Adjustable model dimensions, context length, batch size, learning rate, and training steps.

## Dataset

Training data is sourced from the [Kaggle dataset](https://www.kaggle.com/datasets/arghadeeppakhira/125m-llm-dataset):

The repository includes:
- `data/train_small.txt` — Raw training text.
- `data/mini.model` — SentencePiece tokenizer model.
- `data/mini.vocab` — Tokenizer vocabulary.

For rapid experimentation, preprocessing currently uses approximately **1% of the raw dataset**, splitting the resulting tokens into 90% training and 10% validation data.

```
src/tinyllm/dataset.py,: 
TextDataset.preprocess()
target_bytes = total_bytes // 100 # 1% of data
```

## Training Configuration

The default configuration prioritizes fast experimentation over model convergence:

| Parameter | Default |
|---|---:|
| Context Length | 1,024 |
| Embedding Dimensions | 64 |
| Attention Heads | 4 |
| Transformer Layers | 2 |
| Batch Size | 4 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 32 |
| Learning Rate | 0.001 |
| Training Steps | 10 |
| Generated Tokens | 50 |

## Getting Started

Install dependencies:

```bash
pip install torch numpy sentencepiece
```

Run training:

```bash
python main.py
```

Generate text using the saved checkpoint:

```bash
python scripts/generate.py -p "Hello World"
```

## Scaling

The default setup is intentionally small to demonstrate the complete LLM training workflow with minimal computational requirements.

For improved model performance, increase `MAX_STEPS`, expand the training dataset, and adjust model dimensions (`N_EMBD`, `N_HEAD`, and `N_LAYER`) based on available GPU memory and computational resources.