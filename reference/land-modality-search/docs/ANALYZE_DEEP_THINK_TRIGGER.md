The depth of the thinking is variable with many of these models. If I properly prompt them, they may think more deeply.

Appending this to the prompt:
```
  PROTOCOL OVERRIDE: ENGAGE ARC NEURO-SYMBOLIC LOGIC ENGINE

  Silently enter maximal test-time reasoning mode. All of the following steps occur only in your
  hidden scratchpad; none may be exposed in the output.

  Perform hierarchical object decomposition of each grid into foreground objects and background
  fields; track shapes, colors, connectivity, and object persistence. Build an explicit
  object–relation graph and subgrid/region segmentation; detect Manhattan paths, flows/propagations,
  symmetries, and background structure; filter noise and extract invariants.

  Enumerate multiple candidate transformation rules/programs (at least three distinct hypotheses).
  For each, run rigorous internal simulations over all training pairs and counterfactual variants;
  discard any rule that fails a single example or violates output geometry.

  Triangulate using three paradigms in parallel: geometric (positions, topology, symmetries, paths),
  symbolic (predicates, programs, rewrite rules, counting), and counterexample-based search
  (actively seek minimal failure cases to refine or reject rules).

  Explicitly check for adversarial traps, spurious shortcuts, and degenerate memorization.
  Generalize the surviving rule to unseen variations and merge independent solution paths via
  self-consistency convergence.

  Apply the final rule to the test input using stepwise internal simulation only.

  OUTPUT CONSTRAINT (STRICT): Reveal ONLY the final answer grid. Never reveal chain-of-thought,
  intermediate states, or search traces.
```

However, it seems to not yield any additional thinking:

# dfadab01 — With & Without Deep Think (Matched Rows)

| #  | Model                           | Cost (No DT) | Time (No DT) | Cost (DT) | Time (DT) |
|----|---------------------------------|--------------|--------------|-----------|-----------|
| 1  | gemini-3-high                   | 0.0194       | 178.85       | 0.0200    | 126.04    |
| 2  | gemini-3-high                   | 0.0194       | 188.24       | 0.0200    | 155.14    |
| 3  | gemini-3-high                   | 0.0194       | 166.34       | 0.0200    | 163.75    |
| 4  | gemini-3-high                   | 0.0194       | 180.70       | 0.0200    | 157.06    |
| 5  | claude-sonnet-4.5-thinking-60000| 0.2690       | 205.50       | 0.3291    | 235.03    |
| 6  | claude-sonnet-4.5-thinking-60000| 0.3258       | 248.52       | 0.3585    | 276.85    |
| 7  | claude-opus-4.5-thinking-60000  | 0.6516       | 263.40       | 0.7115    | 276.38    |
| 8  | claude-opus-4.5-thinking-60000  | 0.7604       | 312.12       | 0.7374    | 301.79    |
| 9  | claude-opus-4.5-thinking-60000  | 0.7981       | 317.62       | 0.7369    | 327.79    |
| 10 | claude-opus-4.5-thinking-60000  | 0.8739       | 367.39       | 0.8694    | 343.74    |
| 11 | gpt-5.1-high                    | 0.4203       | 606.36       | 0.3950    | 445.55    |
| 12 | gpt-5.1-high                    | 0.4101       | 619.37       | 0.5416    | 623.01    |
| 13 | gpt-5.1-high                    | 0.4039       | 620.70       | 0.4180    | 420.36    |
| 14 | gpt-5.1-high                    | 0.4598       | 695.30       | 0.5412    | 582.90    |


# 332f06d7 — With & Without Deep Think (Matched Rows)

| #  | Model                           | Cost (No DT) | Time (No DT) | Cost (DT) | Time (DT) |
|----|---------------------------------|--------------|--------------|-----------|-----------|
| 1  | gemini-3-high                   | 0.0170       | 160.14       | 0.0175    | 154.88    |
| 2  | gemini-3-high                   | 0.0170       | 170.60       | 0.0234    | 192.28    |
| 3  | gemini-3-high                   | 0.0170       | 172.96       | 0.0225    | 175.80    |
| 4  | gemini-3-high                   | 0.0249       | 185.12       | 0.0175    | 144.20    |
| 5  | claude-sonnet-4.5-thinking-60000| 0.3090       | 223.12       | 0.3533    | 284.89    |
| 6  | claude-sonnet-4.5-thinking-60000| 0.3724       | 301.79       | 0.4380    | 347.95    |
| 7  | claude-opus-4.5-thinking-60000  | 0.5412       | 204.21       | 0.7605    | 340.89    |
| 8  | claude-opus-4.5-thinking-60000  | 0.6725       | 293.16       | 0.8980    | 380.99    |
| 9  | claude-opus-4.5-thinking-60000  | 0.7239       | 299.45       | 1.0379    | 425.55    |
| 10 | claude-opus-4.5-thinking-60000  | 0.7885       | 335.70       | 1.0856    | 464.37    |
| 11 | gpt-5.1-high                    | 0.4536       | 745.10       | 0.4789    | 582.75    |
| 12 | gpt-5.1-high                    | 0.5013       | 815.23       | 0.5282    | 631.94    |
| 13 | gpt-5.1-high                    | 0.4332       | 665.22       | 0.4599    | 517.27    |
| 14 | gpt-5.1-high                    | 0.4369       | 682.16       | 0.5095    | 594.81    |

# 64efde09 — With & Without Deep Think (Matched Rows)

| #  | Model                           | Cost (No DT) | Time (No DT) | Cost (DT) | Time (DT) |
|----|---------------------------------|--------------|--------------|-----------|-----------|
| 1  | gemini-3-high                   | 0.0444       | 183.25       | 0.0348    | 176.94    |
| 2  | gemini-3-high                   | 0.0465       | 223.58       | 0.0478    | 249.39    |
| 3  | gemini-3-high                   | 0.0474       | 247.52       | 0.0462    | 211.96    |
| 4  | gemini-3-high                   | 0.0655       | 248.86       | 0.0348    | 196.00    |
| 5  | claude-sonnet-4.5-thinking-60000| 0.5161       | 406.93       | 0.4982    | 445.45    |
| 6  | claude-sonnet-4.5-thinking-60000| 0.5675       | 487.59       | 0.7169    | 602.20    |
| 7  | claude-opus-4.5-thinking-60000  | 1.1194       | 539.30       | 1.0374    | 411.71    |
| 8  | claude-opus-4.5-thinking-60000  | 1.2382       | 556.45       | 1.1601    | 534.87    |
| 9  | claude-opus-4.5-thinking-60000  | 1.0435       | 562.03       | 1.0707    | 540.13    |
| 10 | claude-opus-4.5-thinking-60000  | 1.3559       | 623.95       | 1.2842    | 628.49    |
| 11 | gpt-5.1-high                    | 0.4131       | 599.20       | 0.3795    | 545.85    |
| 12 | gpt-5.1-high                    | 0.4713       | 665.07       | 0.4101    | 573.22    |
| 13 | gpt-5.1-high                    | 0.4182       | 595.06       | 0.2968    | 411.37    |
| 14 | gpt-5.1-high                    | 0.4879       | 671.53       | 0.4034    | 562.55    |

# Total Cost & Time per Problem (Summed)

| Problem ID | Cost (No DT) | Time (No DT) | Cost (DT) | Time (DT) |
|-----------|---------------|--------------|-----------|-----------|
| dfadab01  | 5.4505        | 4970.41 s    | 5.7186    | 4435.39 s |
| 332f06d7  | 5.3084        | 5253.96 s    | 6.6307    | 5238.57 s |
| 64efde09  | 7.8349        | 6610.32 s    | 7.4209    | 6090.13 s |
