import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import os
import matplotlib.patheffects as path_effects
from src.tasks import Task
from typing import List, Tuple
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Define a consistent color map for ARC tasks
CMAP = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
NORM = colors.BoundaryNorm(list(range(11)), CMAP.N)
CELL_PIXELS = 15
DPI = 100
PADDING_FACTOR = 1.2

def _draw_cartoon_grid(ax, grid, title):
    rows, cols = grid.shape
    ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest', zorder=0, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)

    # Custom path effects for borders
    border_width = 1
    border_color = 'black'
    effects = [path_effects.withStroke(linewidth=1.5, foreground='w')]

    for r in range(rows + 1):
        for c in range(cols):
            if r == 0 or r == rows or (r < rows and grid[r-1, c] != grid[r, c]):
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], 
                        color=border_color, lw=border_width, zorder=10, path_effects=effects)
    for c in range(cols + 1):
        for r in range(rows):
            if c == 0 or c == cols or (c < cols and grid[r, c-1] != grid[r, c]):
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], 
                        color=border_color, lw=border_width, zorder=10, path_effects=effects)

def generate_and_save_image(task: Task, output_path: str) -> str:
    """
    Generates and saves a cartoon-style visualization of the training pairs for a given ARC task.
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_tasks = [(np.array(ex.input), np.array(ex.output)) for ex in task.train]
    num_pairs = len(train_tasks)
    
    height_ratios = [max(i.shape[0], o.shape[0]) for i, o in train_tasks]
    total_height_cells = sum(height_ratios)
    max_width_cells = max([i.shape[1] + o.shape[1] for i, o in train_tasks])

    fig_height_px = total_height_cells * CELL_PIXELS * PADDING_FACTOR
    fig_width_px = max_width_cells * CELL_PIXELS * PADDING_FACTOR
    
    with plt.xkcd():
        fig = plt.figure(figsize=(fig_width_px / DPI, fig_height_px / DPI))
        fig.patch.set_facecolor('#F8F8F4')
        gs = fig.add_gridspec(num_pairs, 2, height_ratios=height_ratios, hspace=0.5)

        for i, (input_grid, output_grid) in enumerate(train_tasks):
            ax_in = fig.add_subplot(gs[i, 0])
            _draw_cartoon_grid(ax_in, input_grid, f"Input {i+1}")
            ax_out = fig.add_subplot(gs[i, 1])
            _draw_cartoon_grid(ax_out, output_grid, f"Output {i+1}")

            # Add arrow between input and output
            fig.canvas.draw()
            y_pos = (ax_in.get_position().y0 + ax_in.get_position().y1) / 2
            plt.annotate(
                '',
                xy=(ax_out.get_position().x0 - 0.02, y_pos),
                xycoords='figure fraction',
                xytext=(ax_in.get_position().x1 + 0.02, y_pos),
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=12, headwidth=25, lw=3)
            )

        gs.tight_layout(fig, pad=1.0)
        plt.savefig(output_path, dpi=DPI)
        plt.close(fig)

    return output_path
