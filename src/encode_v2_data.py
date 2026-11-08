from pathlib import Path

import torch
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('data/tokenizer_v2.json')

for split in ['train', 'val']:
    text = Path(f'data/{split}_v2.txt').read_text(
        encoding = 'utf-8'
    )

    ids = tokenizer.encode(text).ids
    tokens = torch.tensor(ids, dtype = torch.long)
    torch.save(tokens, f'data/{split}_v2.pt')
    print(split,
          'Символы: ', len(text),
          'Токенов: ', len(tokens),
          'Форма: ', tokens.shape
          )
