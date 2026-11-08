"""Быстрая проверка модели без корпуса и сохранённых весов."""
import torch
from model import LanguageModel

torch.manual_seed(42)
model = LanguageModel(32, 16, 4, 8, 2)
x = torch.randint(0, 32, (2, 8))
logits = model(x)
assert logits.shape == (2, 8, 32)
changed = x.clone()
changed[:, 4:] = (changed[:, 4:] + 1) % 32
with torch.no_grad():
    torch.testing.assert_close(logits[:, :4], model(changed)[:, :4])
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), x.reshape(-1))
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('OK: форма logits, causal mask, конечные градиенты (CPU).')
