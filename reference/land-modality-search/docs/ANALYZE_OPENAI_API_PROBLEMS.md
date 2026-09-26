# Conclusion

# Background
For the hardest problems, 72% (83 out of 116) of the API calls to OpenAI fails. The failures brake down into three categories:

~20% of errors: `Error during execution: OpenAI Job failed: OpenAI Server Error: Code: server_error, Message: An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID wfr_019b5b66259b7a73b2ece70572358eed in your message..`

~40% of errors: `Error during execution: OpenAI Job failed: Timeout after 3600s.`

~40% of errors: `Error during execution: OpenAI Job failed: Token limit: IncompleteDetails(reason='max_output_tokens').`

In my experience, the timeouts end up being one of the other two errors even if settings a 7200s timeout. Hence, we have a real problem. And, a good chunk of it is also indetermined (e.g. "contact OpenAI support", which I've done and they have successfully ignored me, as well as given me completely irrelevant and off-the-shelf-generalized suggestions on how their API works). => We're on our own

# Attempts and ways of relieving the problem

Research suggests a ton of things, like:
- Reducing reasoning level -> can't do that, we're pushing the limit
- Set max output tokens -> I've put it at 120,000, well below the 128,000 limit and it doesn't help
- Remove internal reasoning tracing -> Done that, doesn't matter
- Change the grid representation to "111\222\333\" -> Can't do that, makes the models dumber, see other analysis
- Don't use JSON -> Already not doing that, see other analysis, makes the models dumber
- Verbosity -> Doesn't matter, it's in the reasoning not the actual output
- Truncation parameter -> Only solves for the context window (400,000), that's not the problem
- Prompt adjustments -> All would just make it dumber
- MAYBE: Ask in prompt to output before hitting cap -> Probably doesn't work, and would require some streaming of partial output / reasoning to trigger the model to take different action
- GOOD: Two stage call -> This is very good, effectively doubles the output tokens (256k) if the split can be made clean with a good handover

# Two stage prompts

This will drop performance due to the handover, but it also doubles the reasoning token cap. As long as the added reasoning outweighs "the cost" of doing the handover this might outperform. The best alternative may even be to downlevel to "high" so the performance bar is fairly low.

Prompt 1:

```
Below is an ARC AGI task. You're given the training input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.

Solved examples:
...

Input-only training data:
...

**Goal:** Analyze the input/output pairs to identify the underlying transformation logic.
**Task:** Do not narrow down to a single definitive rule immediately if there is ambiguity. Instead, output a **Prioritized Plan** containing multiple potential transformation hypotheses or edge-case handling strategies.
**Output Constraint:** Output ONLY the list of hypotheses/strategies in natural language. DO NOT write any Python code.
```

Prompt 2 (example):

```
Below is an ARC AGI task. You're given the training input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.

Solved examples:
...

Input-only training data:
...

Here are the potential transformation hypotheses and strategies identified by the Analyst:

1) **Primary hypothesis (most consistent): “Divider + header-driven stamping”**
- Find the solid vertical divider column (a constant non‑zero value repeated down the grid: 2 in ex1, 3 in ex2, 1 in ex3).
- The region to the right of the divider contains a “header/footer” row pattern (top and bottom rows are the same).
- For each interior row, read the **key color** in the cell immediately to the right of the divider (the first column of the right region). When that key color is a “signal” (i.e., not the usual/base color seen there), **stamp that color into the right interior** at the column position(s) where the same color occurs in the header row.
- The stamp shape appears to be a small neighborhood around each header occurrence (often a 3×3 block or at least horizontal ±1 and vertical ±1 around the target), producing the filled clusters seen in the outputs.

2) **How to choose “signal” vs “base” key colors**
- Treat the most frequent key color in that “key column” (first column right of divider) as the default/background for instructions (e.g., 3 in ex1, 4 in parts of ex2, 2 in ex3), and only act on rows where the key differs.
- Alternative if frequency is ambiguous: treat any key color that also appears in the header as actionable; ignore keys not present in the header.

3) **Stamp geometry variants to consider (edge-case handling)**
- Variant A: full **3×3 block** centered at (row r, headerCol c) for each match.
- Variant B: **cross/plus** (center + N/S/E/W).
- Variant C: **horizontal triple** (c−1,c,c+1) on the same row only, with optional vertical propagation to adjacent rows if the examples demand it.
- Clip stamps at boundaries and avoid overwriting divider/border.

4) **Preserve the right-side border structure**
- Keep the divider column unchanged.
- Keep the top and bottom rows unchanged (they act as the reference/header pattern).
- Keep the far-right boundary column values as in input (in examples it remains consistent and often echoes the key color).

5) **Left-of-divider treatment hypotheses (less consistent across examples)**
- Likely action: **erase/zero out everything left of the divider** (matches ex1 & ex3 strongly; ex2 mostly, except the diagonal 8s already happen to remain unchanged).
- Backup strategy: erase all non-zero left-of-divider pixels **except** a specific “special marker object” (e.g., a unique color or a diagonal trio) if it appears to be intentionally preserved; only preserve if removing it would eliminate the only occurrence of that color in the whole grid.

6) **Row-level “do nothing” exceptions**
- If a row’s key color equals the base/default key color, do not stamp anything for that row (matches rows that remain empty in the interior).
- If the header contains multiple occurrences of a color, stamp at **all** of those columns (ex2’s repeated 7s suggests this).

7) **Robustness checks for generalization**
- Confirm divider by searching for a column with the strongest “constant non-zero down the column” signature.
- Confirm header pattern by checking whether first and last rows match (or are near-identical) on the right side; if not identical, prioritize using the top row as the header reference.

**Your Task:**
1. Your task is to write a python function solver(input) that returns the output grid. The solver() function must solve all the input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.
2. Implement the correct logic into a Python function named `solver(input)`.
3. Return only the Python code.
```

The idea here is to split the burden between the two prompts. We want much reasoning to happen in the first prompt, but not all reasoning. I believe the cut above is a good first attempt at splitting this reasonably well.

Let's start off by measuring the relative performance. We already have the following V1B performance (gpt-5.2-low, using problems 247ef758:1,31f7f899:1,7c66cb00:1,136b0064:1,16de56c4:1,36a08778:1,1818057f:1,38007db0:2,bf45cf4b:1,b0039139:2,1ae2feb7:1,7ed72f31:2,b5ca7ac4:1):
- V1B: 1818057f,bf45cf4b,7c66cb00
- V1B: 1818057f,31f7f899,bf45cf4b,b0039139
- V1B: 1818057f,38007db0,136b0064
- V1B: 1818057f,1ae2feb7
- V1B: 1ae2feb7,1818057f,38007db0,31f7f899,247ef758
- V1B: 1818057f,247ef758,7c66cb00
- V1B: 1818057f,247ef758,bf45cf4b
- V1B: 1818057f,1ae2feb7,16de56c4
- V1B: 1818057f,31f7f899,136b0064
- V1B: 1818057f,247ef758
- V1B: 1818057f,247ef758,b0039139
- V1B: 1818057f,bf45cf4b

Let's do the same for the two stage solver:
- V3: 1818057f
- V3: 1818057f,31f7f899
- V3: 1818057f
- V3: 1818057f
- V3: 1818057f
- V3: NA
- V3: NA
- V3: NA
- V3: 1818057f
- V3: 1818057f
- V3: 1818057f
- V3: 1818057f

So, the two-stage solver is significantly under-performing.

# Can the two-stage solve problems that otherwise hit token cap?

These problems consistently fail: a25697e4:1,abc82100:1,e12f9a14:2,269e22fb:2

Outcome when running through 2-stage prompt:
- Run 1: 269e22fb:2=Ok, abc82100:1=Tokens(1), e12f9a14:2=Timeout(1), abc82100:1=Tokens(2)
- Run 2: a25697e4:1=Ok, abc82100:1=Tokens(1), e12f9a14:2=Timeout(2), 269e22fb:2=Support(2)

It seems that it is maybe getting through a bit more, but it's still failing. And, it's failing in both stage 1 and stage 2

