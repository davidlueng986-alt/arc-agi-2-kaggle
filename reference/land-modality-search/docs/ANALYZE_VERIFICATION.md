# gpt-5.1-none
We already know gpt-5.1-none is bad at following instructions. Still, we'll use a simple holding back of one of the training cases to  attempt at verifying the proposed solution. We'll solve the full problem and extract a strategy and then use this on each of the held back training cases.

I've constructed a set of ten problems that I'll use to evaluate the accuracy of the verifications:
```
"tasks": [                                                                                          │
│  3     "data/arc-agi-2-training/1bfc4729.json",                                                          │
│  4     "data/arc-agi-2-training/1d0a4b61.json",                                                          │
│  5     "data/arc-agi-2-training/025d127b.json",                                                          │
│  6     "data/arc-agi-2-training/00576224.json",                                                          │
│  7     "data/arc-agi-2-training/0d3d703e.json",                                                          │
│  8     "data/arc-agi-2-training/19bb5feb.json",                                                          │
│  9     "data/arc-agi-2-training/1a2e2828.json",                                                          │
│ 10     "data/arc-agi-2-training/017c7c7b.json",                                                          │
│ 11     "data/arc-agi-2-training/08ed6ac7.json",                                                          │
│ 12     "data/arc-agi-2-training/0607ce86.json"                                                           │
│ 13   ]
```

The prompt used to generate the strategy is: `Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data.`

Result from running the ten tasks about through ten iterations:
- Pass Rate: 88.00% (88/100)
- Verified Rate: 38.00% (38/100)
- Verified but Failed: 1

The verified but failed is 08ed6ac7, which only has two test cases and thereby susceptible to verification misalignment.

It seems that verifications are often failing despite the problem actually passing. I'll try relaxing the passing logic to "all test cases or at least three test cases passing":
- Pass Rate: 87.00% (87/100)
- Verified Rate: 38.00% (38/100)
- Verified but Failed: 0

It seems that didn't help at all. This surprises me.

Below is the distribution of test cases for all the data:

| Num Examples | Count | Percentage |
|--------------|-------|------------|
| 2            | 158   | 15.80%     |
| 3            | 575   | 57.50%     |
| 4            | 189   | 18.90%     |
| 5            | 49    | 4.90%      |
| 6            | 18    | 1.80%      |
| 7            | 8     | 0.80%      |
| 8            | 2     | 0.20%      |
| 10           | 1     | 0.10%      |

There is a lot of test cases with only two examples. And the bulk has three examples. Let me instead change the logic to this:
- All test cases need to pass
- If a test cases fails, retry it once and if it passes then it's ok
- If there are only two test cases for any problem, then rerun each of them at least one time. The passing criteria becomes to pass each of the two test cases twice, with up to two failures allowed for each of the two cases.

Pseudo code implementation:
```
          * Standard Mode (`len > 2`):
               * Iterate through each training example.
               * Attempt 1: Run the model. If it matches output, mark as passed.
               * Retry Logic: If Attempt 1 fails, run the model again (Attempt 2). If Attempt 2 matches
                 output, mark as passed.
               * Failure: If both attempts fail, the entire verification fails immediately (verified =
                 False, break loop).
           * Two-Example Mode (`len == 2`):
               * Iterate through each of the 2 training examples.
               * For each example, we require 2 successes.
               * We allow up to 2 failures before giving up.
               * Loop: Run the model up to 4 times (max).
                   * Count successes.
                   * Stop as soon as successes == 2 (Pass for this example).
                   * Stop as soon as failures > 2 (Fail for this example -> Verify Fails).
               * If either example fails to reach 2 successes within the allowed margin, the entire
                 verification fails.
```

Run 1:
- Pass Rate: 86.00% (86/100)
- Verified Rate: 40.00% (40/100)
- Verified but Failed: 2 (0d3d703e, 0d3d703e)

Run 2:
- Pass Rate: 88.00% (88/100)
- Verified Rate: 48.00% (48/100)
- Verified but Failed: 3 (017c7c7b, 08ed6ac7, 017c7c7b)

This barely improved the (correct) verification rate, and introduced significant failures. Or put differently, the recall is low (48/88 = 55%) with a non-perfect precision (48-3 / 48 = 94%).

We should optimize for precision to be ~100% for this problem, as we have many other venues to clean out bad solutions. Only later on when we get to close-to-unsolvable problems should we start trading precision for recall.

Let's move to this algorithm:
- All tests must pass
- For any problems with only 2 test cases, run those tests twice - effectively creating four test cases, all which have to pass

A run on this show the following results:
- Pass Rate: 84.00% (84/100)
- Verified Rate: 45.00% (45/100)
- Verified but Failed: 3 (08ed6ac7, 19bb5feb, 08ed6ac7)

19bb5feb even has three test examples. This is likely due to the "strategy" not being well done. We need to dive into the exact examples, with the extracted strategy to understand this better. Let's first verify if this problem also holds true for gpt-5.1-low.

The problems below are "easy for low" but "hard for none":
```
0962bcdd
0c786b71
0ca9ddb6
1190e5a7
1e81d6f9
178fcbfb
0520fde7
1f642eb9
17cae0c1
1e0a9b12
```

The results from running on these are:
- Pass Rate: 85.00% (85/100)
- Verified Rate: 30.00% (30/100)
- Verified but Failed: 2 (178fcbfb, 1f642eb9)

Same problem also on low. I need to understand what's wrong with the "strategy" for these problems.

Looking at the strategys they are wrong, and sort of by mistake still gets the answer right. It's hard to understand why. I think it's just that if you throw enough data at these models, they'll randomly get it right sometimes even with bad instructions. Not sure it'll be possible to raise the precision above the current ~94%.

Also, 94% precision is one way of viewing this. Looking at the sample though, there are only 12-16 failing tasks, out of which 3 are being marked as "verified". That's 20% of the problems being marked as verified when they actually were wrong. We need to run this on a representative data sample to see how bad it actually is

gpt-5.1-none on first 100:
- Pass Rate: 9.62% (20/208)
- Verified Rate: 2.88% (6/208)
- Verified but Failed: 0

gpt-5.1-low on first 100:
- Pass Rate: 38.46% (40/104)
- Verified Rate: 13.46% (14/104)
- Verified but Failed: 2 (05269061, 195ba7dc)

This is really not looking great. There are too many "verified but failed". One pattern I see is the training examples being explicitly explained in the strategy, which of course leads to this. I've fixed this by updating the prompt to `Explain the strategy you used in broad terms such that it can be applied on other similar examples and other input data.` but I still feel this isn't as robust as I need it to be. Having false positives slip through this verification will significantly hurt the overall results.

Idea: Run the solutions twice with strategy retrieval. If they do not output the same grid, then verification failed (and the actual answer). If both succeed, we do backtesting through the training examples using the first of the strategies (don't want to do another call to merge them).

gpt-5.1-none on top 10:
- Pass Rate: 89.00% (89/100)
- Verified Rate: 24.00% (24/100)
- Verified but Failed: 0

gpt-5.1-low on top 10 (low_top_10_low_vs_non):
- Pass Rate: 90.00% (9/10)
- Verified Rate: 50.00% (5/10)
- Verified but Failed: 0

TODO: I really should instead train a model to predict the likelihood of there being a true PASS based on the data supplied (testing from different models, multiple tests, backtesting, etc).

Either way, I'm going to assume that the current approach with two matching runs plus backtesting raises the precision sufficiently high to not harm things. Below are some results that suggests this probably at least fairly true (on representative data):

gpt-5.1-none on first 100:
- Pass Rate: 9.62% (20/208)
- Verified Rate: 1.44% (3/208)
- Verified but Failed: 0

gpt-5.1-low on first 100:
- Pass Rate: 33.65% (35/104)
- Verified Rate: 17.31% (18/104)
- Verified but Failed: 1 (0b148d64)








