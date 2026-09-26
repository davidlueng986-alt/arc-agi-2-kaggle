# Strategy Extraction Analysis

In order to refine the results we need to introduce a concept of "strategy" (or explanation) of why the model has chosen to come to a certain answer/conclusion. This "strategy" works like a guide that can be applied to other test cases thereby enabling us to validate responses, either through testing it on the supplied test data or on synthetic data.

I tested several approaches of getting a "strategy" out of the model. Many had an actual performance implication by affecting the reasoning, through a distraction or deterioration of the spacial reasoning. In the end, the best proved to be a two stage prompt approach where the first stage outputs the solution, and a second stage outputs the strategy with as much context as possible retained from the first step (same session id, etc).

## Approaches Evaluated

### 1. Structured Output (JSON/XML)
We attempted to force the model to output both an explanation and the grid in a single response, structured as JSON or XML.
- **Result:** Significant performance regression (lower PASS rate).
- **Hypothesis:** "Spatial Representation Collapse". Formatting the grid into a JSON string (escaped newlines) or XML structure consumes context and cognitive budget on syntax rather than spatial reasoning. The visual alignment of the grid in the token stream is lost or obfuscated.

### 2. Native Reasoning Summary (OpenAI Specific)
We utilized the `reasoning_effort` parameter combined with `summary: "detailed"` available in newer OpenAI models via the Responses API.
- **Mechanism:** The model generates hidden reasoning tokens and a side-channel summary, while the main content output remains just the grid.
- **Result:** Good performance (doesn't hurt the grid generation), but the summaries are sometimes too abstract or disconnected from the final output generation step if the model decides to pivot silently.

### 3. Two-Stage Prompting (Explain -> Solve)
We prompted the model to first analyze and explain the strategy, and then in a second turn (linked via session ID) asked it to solve the grid.
- **Result:** Mixed. While the explanation was high quality, forcing the model to articulate the strategy *before* solving sometimes led to "hallucinated complexity" or errors where the model talked itself into a wrong corner before generating the grid. The execution often failed even if the explanation seemed plausible.

### 4. Two-Stage Prompting (Solve -> Explain) [Winner]
We prompted the model to solve the grid first (standard text output, high performance), and then immediately followed up with a second prompt in the same session asking for an explanation of the strategy used.
- **Mechanism:** Using `previous_response_id` (OpenAI) to maintain the full hidden reasoning context of the first step.
- **Result:** Optimal. The grid generation is unencumbered (pure text, high pass rate). The explanation is generated *after* the fact but is grounded in the actual reasoning trace that produced the grid. This provides a valid "strategy" without degrading the primary metric.

## Conclusion
The **Solve-Then-Explain** (Reverse Two-Stage) strategy is the recommended approach. It decouples the reasoning load from the formatting load, ensuring the critical "solve" step happens in the most favorable conditions (pure text), while still capturing the necessary metadata for downstream validation.