
import os
import numpy as np
import torch

class TextDataset:
    def __init__(
        self,
        train_path,
        val_path,
        raw_path,
        tokenizer,
        block_size,
        device
    ):
        self.block_size = block_size
        self.device = device

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            self.preprocess(raw_path, train_path, val_path, tokenizer)

        self.train_data = np.memmap(
            train_path, dtype=np.uint16, mode="r"
        )

        self.val_data = np.memmap(
            val_path, dtype=np.uint16, mode="r"
        )

    def preprocess(self, raw_path, train_path, val_path, tokenizer):
        os.makedirs(os.path.dirname(train_path), exist_ok=True)


        total_bytes = os.path.getsize('data/train_small.txt')
        target_bytes = total_bytes // 100
        # total_bytes = os.path.getsize(raw_path)
        # target_bytes = total_bytes // 100

        tokens = []
        bytes_read = 0

        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line_bytes = len(line.encode("utf-8"))

                if bytes_read + line_bytes > target_bytes:
                    break

                tokens.extend(tokenizer.encode(line))
                bytes_read += line_bytes

        split_idx = int(len(tokens) * 0.9)

        train_ids = np.array(tokens[:split_idx], dtype=np.uint16)
        val_ids = np.array(tokens[split_idx:], dtype=np.uint16)

        train_ids.tofile(train_path)
        val_ids.tofile(val_path)

    def get_batch(self, split="train", batch_size=4):
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= self.block_size:
            raise ValueError("Dataset is too small for the configured block size.")

        ix = torch.randint(
            len(data) - self.block_size,
            (batch_size,)
        )

        x = torch.stack([
            torch.from_numpy(
                data[i:i+self.block_size].astype(np.int64)
            )
            for i in ix
        ])

        y = torch.stack([
            torch.from_numpy(
                data[i+1:i+self.block_size+1].astype(np.int64)
            )
            for i in ix
        ])

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y
