"""Shared figure style for the report: same colours for the same thing everywhere."""
import matplotlib.pyplot as plt

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# One fixed colour per model (validated colour-blind-safe categorical order)
MODEL_COLORS = {
    "resnet18": "#2a78d6",            # blue
    "mobilenet_v3_small": "#eb6834",  # orange
    "custom_cnn": "#1baf7a",          # aqua
}
MODEL_LABELS = {
    "custom_cnn": "Custom CNN",
    "mobilenet_v3_small": "MobileNetV3-small",
    "resnet18": "ResNet18",
}
# curve plots: train is neutral grey, validation wears the model's own colour
TRAIN_COLOR = "#8a8984"


def apply_style(grid: bool = True):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.grid": grid,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 10,
        "legend.labelcolor": TEXT_2,
    })
