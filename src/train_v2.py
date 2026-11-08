import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from model import LanguageModel
from pathlib import Path

experiment_dir = Path('experiments/bpe_v2')
experiment_dir.mkdir(parents = True, exist_ok = True)

train_data = torch.load('data/train_v2.pt', weights_only = True)
val_data = torch.load("data/val_v2.pt", weights_only=True)
tokenizer = Tokenizer.from_file('data/tokenizer_v2.json')
vocab_size = tokenizer.get_vocab_size()

total_steps = 5000
batch_size = 4
block_size = 256
embedding_dim = 128
num_heads = 4
num_layers = 6
best_val_loss = float("inf")
best_step = 0
if "LLM_SEED" in os.environ:
    torch.manual_seed(int(os.environ["LLM_SEED"]))

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data, batch_size, block_size):
    starts = torch.randint(0, len(data) - block_size, (batch_size,)).tolist()

    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])

    return x, y


criterion = nn.CrossEntropyLoss()
model = LanguageModel(vocab_size,
                      embedding_dim,
                      num_heads,
                      block_size,
                      num_layers
                      ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

start_step = 0
checkpoint_path = experiment_dir / 'last_checkpoint.pt'

if checkpoint_path.exists():
    checkpoint = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_step = checkpoint['step']
    best_step = checkpoint['best_step']
    best_val_loss = checkpoint['best_val_loss']

    print("Обучение восстановлено с шага", start_step)

def estimate_loss(data, eval_iters = 20):
    losses = []
    model.eval()
    with torch.no_grad():
        for i in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size)
            x_val = xb.to(device)
            y_val = yb.to(device)
            logits_val = model(x_val)
            predictions_val = logits_val.reshape(-1, vocab_size)
            targets_val = y_val.reshape(-1)
            loss_val = criterion(predictions_val, targets_val)
            losses.append(loss_val.item())
    avg_loss = torch.tensor(losses).mean().item()
    return avg_loss

for step in range(start_step ,total_steps):
    xb, yb = get_batch(train_data, batch_size, block_size)
    xb = xb.to(device)
    yb = yb.to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(xb)
    predictions = logits.reshape(-1 ,vocab_size)
    targets = yb.reshape(-1)
    train_loss = criterion(predictions, targets)
    train_loss.backward()
    optimizer.step()

    if (step + 1) % 50 == 0:
        train_loss = estimate_loss(train_data)
        val_loss = estimate_loss(val_data)

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_step = step + 1

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step + 1,
            'best_step': best_step,
            'best_val_loss': best_val_loss,
            'vocab_size': vocab_size,
            'block_size': block_size,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'num_layers': num_layers
        }
        torch.save(checkpoint, experiment_dir/ 'last_checkpoint.pt')

        if is_best:
            torch.save(checkpoint, experiment_dir / 'best_checkpoint.pt')
        print('Шаг', step + 1,
            'Ошибка обучения', train_loss,
            'Ошибка валидации', val_loss,
            'Лучший шаг', best_step,
            'Лучший Loss', best_val_loss)
