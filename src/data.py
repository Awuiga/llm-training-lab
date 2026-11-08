import os

from pathlib import Path

import torch
import torch.nn as nn

from model import LanguageModel


experiment_dir = Path('experiments/char_v1')
experiment_dir.mkdir(parents=True, exist_ok=True)

# Настройки модели
batch_size = 4
block_size = 128
embedding_dim = 128
num_heads = 4
num_layers = 6
if "LLM_SEED" in os.environ:
    torch.manual_seed(int(os.environ["LLM_SEED"]))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Настройки обучения
total_steps = 10000
eval_interval = 50
learning_rate = 0.001


# Загрузка датасета
with open("data/stories.txt", encoding="utf-8") as f:
    text = f.read()


# Создание словаря токенов
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {}
itos = {}

for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch


# Токенизация
encoded = []

for ch in text:
    encoded.append(stoi[ch])

data = torch.tensor(encoded, dtype=torch.long)


# Разделение данных
split_idx = int(len(data) * 0.9)

train_data = data[:split_idx]
val_data = data[split_idx:]


# Создание батчей
def get_batch(data, batch_size, block_size):
    starts = torch.randint(
        0, len(data) - block_size, (batch_size,)
    ).tolist()

    x = torch.stack([
        data[i:i + block_size] for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in starts
    ])

    return x, y


# Создание модели
model = LanguageModel(
    vocab_size,
    embedding_dim,
    num_heads,
    block_size,
    num_layers
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# Оценка качества
def estimate_loss(data, eval_iters=20):
    losses = []
    model.eval()

    with torch.no_grad():
        for i in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size)

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            predictions = logits.reshape(-1, vocab_size)
            targets = yb.reshape(-1)

            loss = criterion(predictions, targets)
            losses.append(loss.item())

    return torch.tensor(losses).mean().item()


# Восстановление обучения
start_step = 0
best_step = 0
best_val_loss = float("inf")

checkpoint_path = experiment_dir / "last_checkpoint.pt"

if checkpoint_path.exists():
    resume_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Проверка совместимости
    expected_config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "block_size": block_size,
        "chars": chars
    }

    for key, expected_value in expected_config.items():
        if resume_checkpoint.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint несовместим: {key}. "
                "Для другой архитектуры нужен отдельный эксперимент."
            )

    model.load_state_dict(
        resume_checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        resume_checkpoint["optimizer_state_dict"]
    )

    start_step = resume_checkpoint["step"]
    best_step = resume_checkpoint["best_step"]
    best_val_loss = resume_checkpoint["best_val_loss"]

    print(f"Обучение восстановлено с шага {start_step}")

else:
    print("Checkpoint не найден. Обучение с нуля.")


# Основной цикл обучения
for epoch in range(start_step, total_steps):
    model.train()

    xb, yb = get_batch(train_data, batch_size, block_size)

    xb = xb.to(device)
    yb = yb.to(device)

    optimizer.zero_grad()

    logits = model(xb)

    predictions = logits.reshape(-1, vocab_size)
    targets = yb.reshape(-1)

    loss = criterion(predictions, targets)

    loss.backward()
    optimizer.step()

    current_step = epoch + 1

    # Валидация и сохранение
    if current_step % eval_interval == 0 or current_step == total_steps:
        train_loss = estimate_loss(train_data)
        val_loss = estimate_loss(val_data)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_step = current_step

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
            "vocab_size": vocab_size,
            "chars": chars,
            "step": current_step,
            "best_step": best_step,
            "best_val_loss": best_val_loss
        }

        # Последнее состояние
        torch.save(checkpoint, experiment_dir / "last_checkpoint.pt")

        # Лучшее состояние
        if improved:
            torch.save(checkpoint, experiment_dir / "best_checkpoint.pt")
            torch.save(
                model.state_dict(),
                experiment_dir / "best_model_weights.pt"
            )

        print(
            "Шаг:", current_step,
            "| Train Loss:", round(train_loss, 4),
            "| Validation Loss:", round(val_loss, 4),
            "| Лучший шаг:", best_step,
            "| Лучший Loss:", round(best_val_loss, 4)
        )


# Итог
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Количество параметров:", total_params)

torch.save(model.state_dict(), experiment_dir / "model_weights.pt")

print("Обучение завершено. Веса сохранены.")
