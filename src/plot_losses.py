"""Построить графики только по восстановленным наблюдениям CSV."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
for version, title in [('char_v1', 'Char v1'), ('bpe_v2', 'BPE v2')]:
    with (root/'reports'/f'{version}.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    fig.patch.set_facecolor('#f8fafc')
    for key, label, color in [('train_loss', 'Train: доступные измерения', '#d97706'), ('val_loss', 'Validation', '#2563eb')]:
        points = [(int(r['step']), float(r[key])) for r in rows if r[key]]
        xs, ys = zip(*points)
        # Разрыв char-кривой нельзя выдавать за непрерывный журнал.
        style = 'o-' if version == 'bpe_v2' and key == 'val_loss' else 'o'
        ax.plot(xs, ys, style, label=label, color=color, markersize=5)
    best = min((r for r in rows if r['val_loss']), key=lambda r: float(r['val_loss']))
    x, y = int(best['step']), float(best['val_loss'])
    ax.annotate(f'Лучший val: {y:.4f}\nШаг {x}', xy=(x,y), xytext=(-105,40), textcoords='offset points', arrowprops={'arrowstyle':'->','color':'#334155'}, fontsize=10)
    ax.set(title=f'{title} — восстановленные измерения', xlabel='Шаг обучения', ylabel='Cross-entropy на токен')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right')
    fig.text(.5, -.035, 'Неполный журнал • без сглаживания • шкалы v1 и v2 напрямую несопоставимы', ha='center', fontsize=9, color='#475569')
    fig.savefig(root/'reports'/f'{version}_loss.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
