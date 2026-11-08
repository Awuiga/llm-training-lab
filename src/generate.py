import os
import torch
from model import LanguageModel

if "LLM_SEED" in os.environ:
    torch.manual_seed(int(os.environ["LLM_SEED"]))

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(os.environ.get("LLM_CHECKPOINT", "experiments/char_v1/best_checkpoint.pt"),
                        map_location = device,
                        weights_only = True)

block_size = checkpoint['block_size']
embedding_dim = checkpoint["embedding_dim"]
num_heads = checkpoint['num_heads']
num_layers = checkpoint['num_layers']



chars = checkpoint['chars']
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

model = LanguageModel(vocab_size,
                      embedding_dim,
                      num_heads,
                      block_size,
                      num_layers
                      ).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

start = 'Однажды мальчик '

tokens = []

for char in start:
    tokens.append(stoi[char])

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)
temperature = 0.7
top_k = 20

with torch.no_grad():
    for i in range(200):
        x_context = x[:, -block_size:]
        logits = model(x_context)
        last_logits = logits[:, -1, :]
        top_values, top_indices = torch.topk(
            last_logits,
            k = top_k,
            dim = -1
        )
        probabilities = torch.softmax(top_values / temperature, dim = -1)
        next_position = torch.multinomial(probabilities, num_samples = 1)
        next_token = torch.gather(top_indices,
                                  dim = 1,
                                  index = next_position)
        x = torch.cat([x, next_token], dim = 1)
generated_text = []
for i in x[0].tolist():
    char = itos[i]
    generated_text.append(char)
generated_text = "".join(generated_text)
print(generated_text)
print("Начальный символ:", start)
