# Grid Format Analysis

This document describes the 9 supported grid representation formats used in prompt engineering.

## 1. Standard
The default format. Rows are separated by newlines, and cells within a row are separated by spaces.
**Example:**
```text
0 1 2
3 4 5
```

## 2. Semicolon
Compact single-line representation. Rows are separated by semicolons `;`.
**Example:**
```text
0 1 2; 3 4 5
```

## 3. XML
XML-style tags wrapping each row.
**Example:**
```xml
<row>0 1 2</row>
<row>3 4 5</row>
```

## 4. CSV
Comma-Separated Values. Standard CSV format.
**Example:**
```text
0,1,2
3,4,5
```

## 5. Python
Standard Python list of lists syntax.
**Example:**
```python
[[0, 1, 2], [3, 4, 5]]
```

## 6. Sparse
List of coordinates and values for non-zero cells. Format: `(row,col):value`.
**Example:**
```text
(0,1):1, (0,2):2, (1,0):3, (1,1):4, (1,2):5
```

## 7. ASCII
Visual representation mapping integers 0-9 to symbols.
Mapping: `0: ., 1: #, 2: x, 3: o, 4: +, 5: *, 6: =, 7: @, 8: %, 9: &`
**Example:**
```text
.#x
o+*
```

## 8. Mask
Decomposed binary masks for each color present in the grid.
**Example:**
```text
Color 1:
0 1 0
0 0 0

Color 2:
0 0 1
0 0 0
```

## 9. Compact
Dense representation. Rows are concatenated strings of digits, separated by `|`.
**Example:**
```text
012|345
```


## Benchmark Results
Model: gpt-5.1-none
Dataset: first_100

| Format | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | Test 6 | Test 7 | Test 8 | Test 9 | Test 10 | Test 11 | Test 12 | Test 13 | Test 14 | Test 15 | Test 16 | Test 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Standard | 9/104 | 8/104 | 8/104 | 11/104 | 9/104 | 10/104 | 11/104 | 8/104 | 8/104 | 9/104 | 9/104 | 9/104 | 10/104 | 9/104 | 8/104 | 12/104 | 10/104 |
| Python | 12/104 | 10/104 | 9/104 | 8/104 | 10/104 | 9/104 | 9/104 | 8/104 | 10/104 | 9/104 | 10/104 | 10/104 | 9/104 | 10/104 | 8/104 | 9/104 | 10/104 |
| Csv | 9/104 | 11/104 | 11/104 | 11/104 | 12/104 | 11/104 | 12/104 | 11/104 | 10/104 | 9/104 | 12/104 | 13/104 | 9/104 | 10/104 | 10/104 | 8/104 | 11/104 |
| Semicolon | 9/104 | 9/104 | 9/104 | 8/104 | 9/104 | 8/104 | 8/104 | - | - | - | - | - | - | - | - | - | - |
| Xml | 7/104 | 8/104 | 9/104 | 10/104 | 8/104 | 10/104 | 10/104 | - | - | - | - | - | - | - | - | - | - |
| Sparse | 9/104 | 6/104 | 7/104 | 6/104 | 6/104 | 6/104 | - | - | - | - | - | - | - | - | - | - | - |
| Ascii | 8/104 | 6/104 | 4/104 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Mask | 8/104 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Compact | 5/104 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

Best format (17 tests): CSV
- Effect size: ≈ 1–1.2 extra solved tasks per 100 compared to Standard and Python.
- Confidence levels (approx):
  - CSV > Standard: ~90%
  - CSV > Python: ~87%
  - CSV best among all three: ~81%

## Sparse Problems Analysis
Pass rates for specific sparse problems across formats (Logs since 11:48:37).

| Format    | 19bb5feb     | 1a2e2828     | 1cf80156     | 1bfc4729     | 1d0a4b61     |
|----------|--------------|--------------|--------------|--------------|--------------|
| Standard  | 100% (6/6)   | 100% (6/6)   | 0% (0/6)     | 100% (6/6)   | 100% (6/6)   |
| Semicolon | 100% (6/6)   | 0% (0/6)     | 0% (0/6)     | 100% (6/6)   | 100% (6/6)   |
| Xml       | 100% (6/6)   | 100% (6/6)   | 0% (0/6)     | 100% (6/6)   | 100% (6/6)   |
| Csv       | 83% (5/6)    | 100% (6/6)   | 0% (0/6)     | 100% (6/6)   | 100% (6/6)   |
| Python    | 100% (6/6)   | 100% (6/6)   | 0% (0/6)     | 100% (6/6)   | 100% (6/6)   |
| Sparse    | 100% (5/5)   | 100% (5/5)   | 80% (4/5)    | 100% (5/5)   | 100% (5/5)   |
| Ascii     | 100% (2/2)   | 100% (2/2)   | 0% (0/2)     | 100% (2/2)   | 50% (1/2)    |

