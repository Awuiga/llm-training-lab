from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size = 2048,
    min_frequency = 2,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
)

tokenizer.train(['data/train_v2.txt'], trainer)

tokenizer.save('data/tokenizer_v2.json')

example = "Однажды мальчик отправился в лес."

ids = tokenizer.encode(example).ids
restored = tokenizer.decode(ids)

print("Размер словаря:", tokenizer.get_vocab_size())
print("ID токенов:", ids)
print("Восстановленный текст:", restored)

assert restored == example
print("Проверка пройдена!")
