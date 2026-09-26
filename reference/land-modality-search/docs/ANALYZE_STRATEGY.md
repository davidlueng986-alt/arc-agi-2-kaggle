
# gpt-5.1-none
It's pretty much impossible to teach it anything. It largely doesn't respond to being supplied with a strategy. I tested a few problems with very explicit solutions, and it didn't pick up on anything.

# gpt-5.1-low
This one seems to respond to explicit strategies, but some are too complicated and hints-only is not working

## 1990f7a8
Strategy: NA
Pass rate: 1/10

Strategy:
```
Each input grid contains four separate 3×3 shapes made of color 2.
I crop each 3×3 shape (keeping its exact pattern), then place them into a new 7×7 grid as:
	•	top‑left: upper-left shape from the input
	•	top‑right: upper-right shape from the input
	•	bottom‑left: lower-left shape from the input
	•	bottom‑right: lower-right shape from the input
with a one‑cell row and column of zeros separating them (so shapes go in the 3×3 corners of the 7×7).
```
Pass rate: 4/10

Strategy: `Move an organize the objects. Create a separator between them.`\
Pass rate: 3/10

Strategy: `Move 3x3 objects to corners with an empty divider between them, creating a 7x7 output grid.`
Pass rate: 4/10

Strategy: `Hint 1: Move objects. Hint 2: Empty divider.`
Pass rate: 1/10

## 10fcaaa3

Strategy: NA
Pass rate: 0/10

Strategy:
```
First, tile the input grid into a 2×2 array of copies, doubling both height and width. Then, for every non-zero cell in this enlarged grid, keep its color, and place color 8 in each of its diagonal neighbors (up-left, up-right, down-left, down-right) as long as they stay within bounds and don’t overwrite a non-zero color.
```
Pass rate: 9/10

Strategy: `copy it, then expand into the corners`
Pass rate: 0/10

Strategy: `Copy four times. Expand into corners.`
Pass rate: 1/10

## 0a2355a6

Strategy: NA
Pass rate: 0/10

Strategy: `Each 8-blob is treated independently. For every 4-connected component of 8s, count how many “holes” (enclosed 0-regions fully surrounded by that component) it contains. Then recolor the entire component with a single color equal to that hole count: 1 → color 1, 2 → color 2, 3 → color 3, 4 → color 4. Background 0s stay 0.`
Pass rate: 0/10

Strategy:
```
1. Work one 8-region at a time
	•	Look at the grid and find each connected group of 8s.
	•	“Connected” means 4-connected: cells are neighbors if they touch up, down, left, or right (diagonals do not count).
	•	Each such group is a component that will be recolored to a single new color.

⸻

2. Define what a “hole” is

We only care about 0-cells (background) that are completely enclosed by one 8-component.

A hole is:
	•	A 4-connected region of 0s
	•	Such that every 4-neighbor of any cell in this 0-region is either:
	•	another 0 in the same region, or
	•	an 8 belonging to this same component

and no cell in the 0-region is connected (via 4-steps through 0s) to:
	•	any 0 on the outer border of the grid, or
	•	any 0 that touches another 8-component.

Intuition:
If you imagine coloring all the 8s in one component and treating everything else as “background,” a hole is a “lake” of 0s fully surrounded by that component — you cannot walk from that 0-region to the edge of the grid using only 0-steps.

⸻

3. How to count holes for one component (algorithmically)

For a given 8-component:
	1.	Mark all its 8-cells.
	2.	Consider the 0-cells that are adjacent to this component (or inside its shape).
	3.	Among those 0-cells, find all maximal 4-connected groups of 0s (using flood fill / BFS / DFS).
	4.	For each such 0-group:
	•	Temporarily ask: “If I start in this 0 and walk only on 0s, can I reach the grid boundary?”
	•	If yes, it is not a hole (it leaks to the outside).
	•	If no, it is a hole.
	5.	The hole count for that component is simply the number of these enclosed 0-groups.

⸻

4. Map “hole count” → new color

Once you have the hole count h for that 8-component:
	•	If h = 1, replace all 8s in that component with color 1.
	•	If h = 2, replace all 8s in that component with color 2.
	•	If h = 3, replace all 8s in that component with color 3.
	•	If h = 4, replace all 8s in that component with color 4.

(Those mappings come directly from the examples: components with 1, 2, 3, 4 holes consistently become colors 1, 2, 3, 4 respectively.)

All 0s stay 0.
```
Pass rate: 0/10


## 11dc524f

Strategy: NA
Pass rate: 0/10

Strategy: `Find the connected cluster of 2s and the cluster of 5s. Translate the 2-cluster in a straight line (horizontally or vertically) until its bounding box is directly adjacent to the 5-cluster’s bounding box. Then erase the original 5s and redraw them as a mirror image of the final 2-shape across the line between the two clusters, so the 2s and 5s form a symmetric pair.`
Pass rate: 9/10

Strategy: `Hint 1: Focus on connected clusters of same colors. Hint 2: Move things around. Hint 3: Create a mirror image.`
Pass rate: 0/10

### Test with successively less specific strategies

Strategy: `Identify the connected cluster of color 2 and the connected cluster of color 5. Determine their bounding boxes. Translate the 2-cluster in a straight line so its bounding box becomes directly adjacent to the 5-cluster. After positioning the 2s, erase the original 5s and redraw a new 5-shape that is the mirror image of the translated 2-shape across the boundary line between them. Leave all other cells as 7.`
Pass rate: 9/10

Strategy: `Locate two distinct non-background color clusters. Move the first cluster so that its bounding box touches the bounding box of the second cluster in either a horizontal or vertical direction. Then remove the second cluster’s original shape and recreate it as a mirrored copy of the moved first cluster, reflected across the line separating their bounding boxes.`
Pass rate: 6/10

Strategy: `When the grid contains two objects, reposition one so that it forms a deliberate adjacency with the other. After aligning them, replace the second object with a reflection of the first, ensuring the two shapes become symmetric partners. The background remains unchanged.`
Pass rate: 7/10

Strategy: `If the task contains two meaningful structures, consider repositioning or transforming one object, then generating a new object by applying a geometric transformation (like reflection or rotation) derived from the first. Many ARC tasks expect you to infer symmetry or alignment relationships between repeated or related shapes.`
Pass rate: 1/10

Strategy: `When faced with a problem that includes multiple elements, look for relationships between them. Try adjusting one element and then apply a consistent transformation to another so they form a coherent pattern. Use symmetry, alignment, or repetition to guide how elements should be repositioned or modified.`
Pass rate: 0/10

# gpt-5.1-medium

It's very good at following an explicit instruction on how to solve the specific problem. In cases where it has no idea how to solve a problem (success 0/10) with a given brief expanation of how to solve it fully succeeds (10/10 success).
It seems to be able of improving with generic advice as long as it's applicable to the problem. Things like:
- Identify structural dividers
- Extract coherent objects
- Cluster objects into a coarse grid
- Identify what counts as background vs. meaningful structure.
- Map legend positions to block positions
- Compress the grid into a higher-level representation
But, it's not smart enough for this to make it able of solving everything reliably. It for sure couldn't be just standardized advice, it would have to be based on insight about the nature of the problem, probably to the extent of specificity that problems need to be put into 30+ different groups.

## 0a1d4ef5

Strategy: NA
Pass rate: 3/10

Strategy:
```
	1.	Find background vs foreground colors
	•	For each color, compute its bounding box.
	•	Any color whose bounding box spans the entire grid (from (0,0) to (H‑1,W‑1)) is treated as background.
	•	All other colors are foreground.
	2.	Extract foreground components
	•	On the grid, ignore background colors and find 4‑connected components of same‑colored foreground pixels.
	•	Each component will correspond to one cell in the final compressed layout.
	•	For each component, record its color and bounding box.
	3.	Group components into rows and columns
	•	Build row clusters: components whose vertical bounding intervals overlap belong to the same logical row of blocks.
	•	Build column clusters: components whose horizontal bounding intervals overlap belong to the same logical column of blocks.
	•	Sort row clusters by their top row, and column clusters by their left column.
	•	The number of row clusters × column clusters gives the output grid size.
	4.	Fill the output grid
	•	For each component, find which row cluster and column cluster its bounding box belongs to; place its color in that (row, col) of the output grid.
	•	Each (row, col) pair has exactly one component.
```
Pass rate: 10/10

Strategy:
```
	1.	Separate background from signal
	•	Identify which values represent “background” (often those that are most common, span the whole area, or form a uniform field).
	•	Treat everything else as “foreground objects.”
	2.	Extract coherent objects
	•	Find connected components of foreground cells (4- or 8-connectivity).
	•	Each component is one logical object; record its color and bounding box.
	3.	Cluster objects into a coarse grid
	•	Along the vertical axis, group objects into rows based on overlapping vertical spans.
	•	Along the horizontal axis, group objects into columns based on overlapping horizontal spans.
	•	Sort row groups by top coordinate, column groups by left coordinate.
	•	The number of row groups × column groups gives the size of the compressed layout.
	4.	Assign one object per cell
	•	For each object, determine which row group and column group it belongs to.
	•	Place a representative value (e.g., the object’s color) into that cell of the output grid.
	5.	Optionally simplify
	•	If adjacent rows or columns are identical or redundant, you can merge them to further compress the pattern.
```
Pass rate: 10/10

Strategy:
```
	1.	Identify what counts as background vs. meaningful structure.
	2.	Group meaningful cells into connected objects.
	3.	Organize those objects into logical rows and columns based on their spatial arrangement.
	4.	Build an output grid where each cell represents one object or cluster.
	5.	**Optionally compress or merge rows/columns when patterns repeat.
```
Pass rate: 8/10

Strategy:
`Hint 1: Identify what counts as background vs. meaningful structure. Hint 2: Group meaningful cells into connected objects.`
Pass rate: 4/10


## 09629e4f

Strategy: NA
Pass rate: 0/10

Strategy:
```
	•	The 11×11 grid is partitioned by rows and columns of 5s into nine 3×3 blocks.
	•	In each example there is exactly one 3×3 block that has no 8s.
	•	That special block acts as a legend:
	•	Each cell (local position) (i,j) in this block corresponds to the big block at position (i,j) in the 3×3 grid of blocks.
	•	If the legend cell is 0, the corresponding big block becomes all 0.
	•	If the legend cell is 2, 3, 4, or 6, the corresponding big block is filled entirely with that color.
	•	The rows and columns of 5s are kept unchanged as separators.
```
Pass rate: 10/10

### 5 steps of successive lowering of specificity

Strategy:
```
	1.	Identify structural dividers
	•	Look for a recurring “frame” color that forms full rows and columns, partitioning the grid into equal sub-blocks.
	•	Treat those lines as fixed separators that stay the same in the output.
	2.	Treat each sub-block as a “cell” in a coarse 3×3 grid
	•	Once partitioned, think of the whole grid as a 3×3 board where each “cell” is itself a 3×3 mini-grid.
	3.	Search for a “control” sub-block
	•	Among the nine sub-blocks, find one whose content is qualitatively different (for example, missing a certain color that appears in all other blocks, or having simpler patterns).
	•	Interpret this unusual block as a legend that encodes how the other blocks should be recolored.
	4.	Map legend positions to block positions
	•	Match each cell position inside the legend (top-left, top-middle, etc.) to the corresponding block in the 3×3 layout.
	•	The value at that legend cell determines the uniform fill color of the corresponding block in the output, while the dividers stay unchanged.
```
Pass rate: 10/10

Strategy:
```
	1.	Detect repeated bands or stripes
	•	Scan for rows or columns that are uniform and repeat periodically.
	•	These usually indicate logical partitions of the grid (like “rows of cells” in a table).
	2.	Compress the grid into a higher-level representation
	•	After finding partitions, conceptually shrink each sub-block (e.g. 3×3 region) into a single abstract “meta-cell”.
	•	The puzzle then becomes transforming a small meta-grid instead of the full detailed one.
	3.	Look for a rule that maps sub-block content → meta-color
	•	For each sub-block, identify simple statistics:
	•	Which colors occur?
	•	Which colors are missing?
	•	Are there special shapes, such as a single pixel, a stripe, a diagonal?
	•	Try to see if any one sub-block acts as a codebook: its internal pattern explains how to recolor or rewrite the others.
	4.	Apply the meta-rule back to the full grid
	•	Once you find the rule on the meta-grid, expand it back so each meta-cell becomes a uniformly colored block (or some simple pattern) in the original resolution.
	•	Preserve any structural elements (dividing lines) that are clearly invariant across examples.
```
Pass rate: 3/10

Strategy:
```
	1.	Compare multiple examples at a bird’s-eye level first
	•	For each example, do NOT get lost in pixel-by-pixel detail at first.
	•	Instead, answer:
	•	What stays exactly the same from input to output?
	•	What changes, and where are those changes concentrated?
	2.	Classify colors into roles
	•	Treat colors not as numbers but as roles: divider, background, object, marker, label, etc.
	•	Ask:
	•	Which color often forms full lines or borders?
	•	Which colors appear in compact clusters?
	•	Which colors seem to “label” regions rather than form shapes?
	3.	Look for a control mechanism
	•	Many ARC tasks have a “control” area: a small region whose pattern dictates what happens elsewhere (e.g., a legend, a prototype, or a set of parameters).
	•	Look for a region with:
	•	Distinct constraints (fewer colors, simpler layout).
	•	A one-to-one correspondence with something in the rest of the grid (e.g. same count of items, matching positions, etc.).
	4.	Propose & test a simple hypothesis
	•	Formulate a candidate rule in plain language:
	•	“Each object is recolored based on the color at the same position in the control region.”
	•	“Each region becomes uniform with the majority color present.”
	•	Check it against all given examples. If it fails anywhere, adjust or discard.
```
Pass rate: 2/10


## 045e512c

Strategy: NA
Pass rate: 0/10

Strategy:
```
	1.	In each example there is one “main” shape: the color with the most cells (anchor).
	2.	Take the union of all cells of this anchor color; that exact pattern is the template shape.
	3.	For every other color, look at each connected component:
	•	Compute the vector from the anchor’s center to the component’s center.
	•	Quantize each component of this vector to -4, 0, or +4 (keeping the sign).
	•	Starting from the anchor’s center, repeatedly step by that vector, and at each step stamp a copy of the anchor’s template shape in the component’s color (truncating at the grid edges).
	4.	The anchor-color cells stay as they are.
```
Pass rate: 7/10


# gpt-5.1-high

## 1190bc91

Strategy: NA
Pass rate: 1/10

Strategy:
```
Across the three examples, the grid always contains:
	1.	One main long line of different colors
	•	Ex1: the long horizontal row 2,3,7,9,5,1,8
	•	Ex2: the long vertical column 8,4,6,3,2,1
	•	Ex3: the vertical 3,2,4
	•	Test: the vertical 2,4,8,6,3 in column 2
	2.	Short constant segments on the border (all same color, length ≥ 2)
	•	Ex1: 6,6 at top, and 4,4 on the left
	•	Ex2: 7,7 at top, and 9,9 on the right
	•	Ex3: 8,8 on the left
	•	Test: 9,9 at top, and 7,7 on the right

The transformation:

1. Diagonal “spray” from the main line
	•	Take the longest line (main spine): in the test, that’s the vertical 2,4,8,6,3 in column 2.
	•	Each cell of this line keeps its color and copies that color out along all four diagonals (↖, ↗, ↙, ↘).
	•	You keep extending along a diagonal until you hit a non-zero cell; you don’t overwrite existing colors.
	•	This builds a kind of diamond of the spine’s colors (just like the 8–4–6–3–2 in Example 2 spread diagonally).

2. Diagonal spray from the other original non-zero cells
	•	Do the same diagonal propagation from the other original non-zero cells (like the top pair and side pair), also stopping as soon as you hit any non-zero cell.

3. Extend constant segments along their row/column

For each constant segment (all same color, length ≥ 2) from the original grid:
	•	If it’s horizontal (like 9,9 in the test on row 0):
	•	Extend that color left and right along the same row into zeros until you hit another non-zero.
	•	If it’s vertical (like 7,7 in the test on column 7, rows 3–4):
	•	Extend that color up and down along the same column into zeros until you hit another non-zero.

This gives you:
	•	A band of 9s across the top (in the test), and
	•	A vertical bar of 7s that stretches along column 7.

4. Extra diagonal spread from the extended constant segments

Finally, from all cells that belong to these extended constant segments:
	•	Spread their color diagonally (↖, ↗, ↙, ↘) again.
	•	While doing this:
	•	You may pass through cells of the same color (so you can “go past” earlier 9s or 7s),
	•	But you stop if you hit a different non-zero color,
	•	And you only fill zeros.

This turns the constant segments into the larger triangles/diamonds of 6s, 7s, 9s, 4s you see in the examples.
```
Pass rate: 8/10


Strategy:
```
Across the three examples, the grid always contains one main long line of different colors, and a few short constant segments on the border (all same color, length ≥ 2)

The transformation:
1. Diagonal “spray” from the main line
2. Diagonal spray from the other original non-zero cells
3. Extend constant segments along their row/column
4. Extra diagonal spread from the extended constant segments

Finally, from all cells that belong to these extended constant segments:
	•	Spread their color diagonally (↖, ↗, ↙, ↘) again.
	•	But you stop if you hit a different non-zero color,
	•	And you only fill zeros.

This turns the constant segments into the larger triangles/diamonds of 6s, 7s, 9s, 4s you see in the examples.
```
Pass rate: 3/10

Strategy: `Extend diagonally for the main line. Then for the rest, fill their area.`
Pass rate: 1/10


