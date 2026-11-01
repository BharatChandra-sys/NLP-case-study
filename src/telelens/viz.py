import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def hbar(path, labels, values, title, xlabel, xlim=None, highlight=None, fmt="{:.3f}"):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(labels) + 1.0))
    colors = [BLUE if highlight is None or label == highlight else "#9ec5f4" for label in labels]
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=colors, height=0.68)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    left, right = ax.get_xlim()
    for row, value in zip(y, values):
        ax.text(value + (right - left) * 0.01, row, fmt.format(value), va="center", fontsize=8, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def lines(path, series, title, xlabel, ylabel, xlim=None, ylim=None):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (name, xs, ys), color in zip(series, SERIES):
        ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def grouped(path, labels, groups, title, ylabel, ylim=None):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / len(groups)
    for i, ((name, values), color) in enumerate(zip(groups, SERIES)):
        xs = [x + i * width - 0.4 + width / 2 for x in range(len(labels))]
        ax.bar(xs, values, width * 0.92, color=color, label=name)
    ax.set_xticks(range(len(labels)), [label.replace(" ", "\n", 1) for label in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=24)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(len(groups), 4))
    fig.savefig(path)
    plt.close(fig)


def pca_words(path, points, title):
    style()
    fig, axes = plt.subplots(1, len(points), figsize=(7.4, 4.2))
    for ax, (name, rows) in zip(axes, points.items()):
        xs, ys = [r[1] for r in rows], [r[2] for r in rows]
        span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
        ax.set_xlim(min(xs) - 0.12 * span_x, max(xs) + 0.28 * span_x)
        ax.set_ylim(min(ys) - 0.08 * span_y, max(ys) + 0.08 * span_y)
        char_w, height = 0.021 * span_x, 0.042 * span_y
        pad_x, pad_y = 0.018 * span_x, 0.018 * span_y
        boxes = [(x - pad_x / 2, y - pad_y, x + pad_x / 2, y + pad_y) for x, y in zip(xs, ys)]
        for word, x, y, polar in sorted(rows, key=lambda r: -r[1]):
            color = {1: BLUE, -1: ORANGE}.get(polar, MUTED)
            ax.scatter(x, y, s=28, color=color, edgecolors=SURFACE, linewidths=1, zorder=3)
            width = len(word) * char_w
            options = [(x + pad_x, y - height / 2), (x - pad_x - width, y - height / 2),
                       (x - width / 2, y + pad_y), (x - width / 2, y - pad_y - height),
                       (x + pad_x, y + pad_y), (x + pad_x, y - pad_y - height),
                       (x - pad_x - width, y + pad_y), (x - pad_x - width, y - pad_y - height)]
            def clashes(left, bottom):
                return sum(left < b[2] and left + width > b[0] and bottom < b[3] and bottom + height > b[1] for b in boxes)
            left, bottom = min(options, key=lambda o: clashes(*o))
            boxes.append((left, bottom, left + width, bottom + height))
            ax.text(left, bottom + height / 2, word, va="center", ha="left", fontsize=6.5, color=INK)
        ax.set_title(name, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Positive log-odds"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Negative log-odds")]
    fig.legend(handles=handles, loc="lower center", ncols=2)
    fig.subplots_adjust(top=0.84, bottom=0.11, wspace=0.05)
    fig.savefig(path)
    plt.close(fig)
