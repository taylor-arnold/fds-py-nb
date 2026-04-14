import numpy as np
import polars as pl
from plotnine import *
from polars import col as c
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
#model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)


# Load and prepare Shakespeare data
plays = pl.read_csv("data/shakespeare_lines.csv.gz")
plays = plays.filter(~c.text.is_null())
corpus = "\n".join(plays["text"].to_list())

# Tokenize using GPT-2's tokenizer (not tiktoken)
all_tokens = tokenizer.encode(corpus)
all_tokens = torch.tensor(all_tokens, dtype=torch.long)
print(f"Total tokens: {len(all_tokens):,}")

# Dataset - same sliding window approach as before
CONTEXT_LENGTH = 256
STRIDE = 128

class ShakespeareDataset(Dataset):
    def __init__(self, tokens, context_length, stride):
        self.tokens = tokens
        self.context_length = context_length
        self.examples = []
        for i in range(0, len(tokens) - context_length, stride):
            self.examples.append(i)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        start = self.examples[idx]
        chunk = self.tokens[start : start + self.context_length]
        # pass the same sequence as both input and labels
        # the model internally shifts for next-token prediction
        return chunk, chunk

# Train/val split
split = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:split]
val_tokens = all_tokens[split:]

train_dataset = ShakespeareDataset(train_tokens, CONTEXT_LENGTH, STRIDE)
val_dataset = ShakespeareDataset(val_tokens, CONTEXT_LENGTH, STRIDE)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Fine-tuning hyperparameters - much smaller LR than training from scratch
EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_STEPS = 50

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS

# Warmup + cosine schedule
def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))

# Training loop
step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        step += 1

        # GPT2LMHeadModel can compute the loss for you
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            avg = running_loss / batch_idx
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} | batch {batch_idx:>5}/{len(train_loader)} "
                  f"| loss {avg:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            val_loss += outputs.loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch}/{EPOCHS} — train loss: {train_loss:.4f}, "
          f"val loss: {val_loss:.4f}")

# Save best model
torch.save(model.state_dict(), 'models/shakespeare_gpt2_finetuned.pt')

# Generate some text to test
model.eval()
prompt = "To be or not to be"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=200,
        temperature=0.8,
        top_k=40,
        do_sample=True,
    )

print("\n--- Generated text ---")
print(tokenizer.decode(output[0]))