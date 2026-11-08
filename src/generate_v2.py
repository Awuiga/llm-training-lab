import os
import torch
import torch.nn as nn
from model import LanguageModel
from tokenizers import Tokenizer


if "LLM_SEED" in os.environ:
    torch.manual_seed(int(os.environ["LLM_SEED"]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(
    os.environ.get("LLM_CHECKPOINT", "experiments/bpe_v2/best_checkpoint.pt"),
    map_location = device,
    weights_only = True
)

tokenizer = Tokenizer.from_file('data/tokenizer_v2.json')

model = LanguageModel(
    checkpoint['vocab_size'],
    checkpoint['embedding_dim'],
    checkpoint['num_heads'],
    checkpoint['block_size'],
    checkpoint['num_layers']
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
prompt = 'Однажды мальчик'
encoded = tokenizer.encode(prompt)
token_ids = encoded.ids

input_ids = torch.tensor([token_ids], dtype = torch.long, device = device)

temperature = 0.65
max_new_tokens = 100
top_k = 40

with torch.no_grad():
    for i in range(max_new_tokens):
        context = input_ids[:, -checkpoint['block_size']:]
        logits = model(context)
        last_logits = logits[:, -1, :]
        top_values, top_indices = torch.topk(
            last_logits,
            k = top_k,
            dim = 1
        )
        probabilities = torch.softmax(top_values / temperature, dim = -1)
        next_position = torch.multinomial(probabilities, num_samples = 1)
        next_token = torch.gather(top_indices,
            dim = 1,
            index = next_position
        )
        input_ids = torch.cat([input_ids, next_token], dim = 1)
generated_ids = input_ids[0].tolist()
generated_text = tokenizer.decode(generated_ids)

print("\nСгенерированная история:")
print(generated_text)
