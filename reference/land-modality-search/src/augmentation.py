from typing import List
import copy

Grid = List[List[int]]

def rotate_grid_90(grid: Grid) -> Grid:
    """Rotates the grid 90 degrees clockwise."""
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0 for _ in range(rows)] for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            new_grid[c][rows - 1 - r] = grid[r][c]
    return new_grid

def rotate_grid_180(grid: Grid) -> Grid:
    """Rotates the grid 180 degrees."""
    # Rotate 90 twice
    return rotate_grid_90(rotate_grid_90(grid))

def rotate_grid_270(grid: Grid) -> Grid:
    """Rotates the grid 270 degrees clockwise (or 90 counter-clockwise)."""
    # Rotate 90 three times
    return rotate_grid_90(rotate_grid_90(rotate_grid_90(grid)))

def flip_grid_horizontal(grid: Grid) -> Grid:
    """Flips the grid horizontally (left <-> right)."""
    if not grid:
        return []
    return [row[::-1] for row in grid]

def flip_grid_vertical(grid: Grid) -> Grid:
    """Flips the grid vertically (up <-> down)."""
    if not grid:
        return []
    return grid[::-1]

def flip_grid_both(grid: Grid) -> Grid:
    """Flips both horizontally and vertically (equivalent to 180 rotation)."""
    return flip_grid_vertical(flip_grid_horizontal(grid))

def shift_grid_colors(grid: Grid, shift: int) -> Grid:
    """Shifts all colors in the grid by 'shift' amount (modulo 10)."""
    if not grid:
        return []
    return [[(cell + shift) % 10 for cell in row] for row in grid]

def get_augmented_pairs(input_grid: Grid, output_grid: Grid) -> List[dict]:
    """
    Generates a list of augmented (input, output, type) tuples.
    Types: rotation_90, rotation_180, rotation_270,
           reflection_h, reflection_v, reflection_both,
           color_shift_1, color_shift_2, color_shift_3
    """
    augmented = []

    # Rotations
    augmented.append({
        "type": "rotation_90",
        "input": rotate_grid_90(input_grid),
        "output": rotate_grid_90(output_grid)
    })
    augmented.append({
        "type": "rotation_180",
        "input": rotate_grid_180(input_grid),
        "output": rotate_grid_180(output_grid)
    })
    augmented.append({
        "type": "rotation_270",
        "input": rotate_grid_270(input_grid),
        "output": rotate_grid_270(output_grid)
    })

    # Reflections
    augmented.append({
        "type": "reflection_h",
        "input": flip_grid_horizontal(input_grid),
        "output": flip_grid_horizontal(output_grid)
    })
    augmented.append({
        "type": "reflection_v",
        "input": flip_grid_vertical(input_grid),
        "output": flip_grid_vertical(output_grid)
    })
    augmented.append({
        "type": "reflection_both",
        "input": flip_grid_both(input_grid),
        "output": flip_grid_both(output_grid)
    })

    # Color Shifts
    for s in [1, 2, 3]:
        augmented.append({
            "type": f"color_shift_{s}",
            "input": shift_grid_colors(input_grid, s),
            "output": shift_grid_colors(output_grid, s)
        })

    return augmented
