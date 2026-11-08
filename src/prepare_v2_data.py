from pathlib import Path
import random

text = Path("data/stories.txt").read_text(encoding="utf-8")

stories = [
    story.strip()
    for story in text.split("\n\n")
    if story.strip()
]

random.Random(42).shuffle(stories)

split_idx = int(len(stories) * 0.9)

train_stories = stories[:split_idx]
val_stories = stories[split_idx:]

Path("data/train_v2.txt").write_text(
    "\n\n".join(train_stories),
    encoding="utf-8"
)

Path("data/val_v2.txt").write_text(
    "\n\n".join(val_stories),
    encoding="utf-8"
)

print("Обучающих рассказов:", len(train_stories))
print("Проверочных рассказов:", len(val_stories))
