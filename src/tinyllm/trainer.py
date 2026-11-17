import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import torch

class Trainer:
    def __init__(
        self,
        model,
        dataset,
        learning_rate,
        grad_accum_steps,
        device
    ):
        self.model = model
        self.dataset = dataset
        self.grad_accum_steps = grad_accum_steps
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        # self.scaler = torch.amp.GradScaler(
        #     "cuda",
        #     enabled=(device == "cuda")
        # )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(device == "cuda")
        )

    def train(self, max_steps, batch_size):
        self.model.train()

        for step in range(max_steps):
            self.optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0

            for _ in range(self.grad_accum_steps):
                xb, yb = self.dataset.get_batch(
                    "train", batch_size
                )

                with torch.autocast(
                    device_type="cuda",
                    enabled=(self.device == "cuda")
                ):
                    _, loss = self.model(xb, yb)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                accumulated_loss += loss.item() * self.grad_accum_steps

            self.scaler.step(self.optimizer)
            self.scaler.update()

            print(
                f"Step {step}: Loss = {accumulated_loss:.4f}"
            )
