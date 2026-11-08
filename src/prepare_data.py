from pathlib import Path

Path("data").mkdir(exist_ok=True)

from datasets import load_dataset

dataset = load_dataset(
    "inkoziev/ru_stories",
    split="train",
    revision="26173ec5e84f118a0294f6cf8936dfef1f39033d",
    streaming=True
)

iterator = iter(dataset)
count = 0

with open('data/stories.txt', 'w', encoding = 'utf-8') as f:

    for story in dataset:
        sentences = [
            story['sentence1'],
            story['sentence2'],
            story['sentence3'],
            story['sentence4'],
            story['sentence5']
        ]

        story = ' '.join(sentences)
        count += 1
        f.write(story + '\n\n')
print(count)
