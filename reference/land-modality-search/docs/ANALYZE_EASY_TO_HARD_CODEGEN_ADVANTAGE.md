# Conclusion

It seems that the approach extends reasonably well to making hard test cases within otherwise reasonable problems solvable.

# Premise

Some problems have easy test examples, as well as very hard test examples. Below are a few of these (solvable using gpt-5.2-low):
- 247ef758 (test 2 ~100th easiest)
- 16de56c4 (test 2 ~70th easiest)
- 36a08778 (test 2 ~40th easiest)
- 38007db0 (test 1/2 ~50th easiest)
- b0039139 (test 1/2 ~20th easiest)
- 1ae2feb7 (test 1/2/3 ~50th easiest)
- 7ed72f31 (test 1 ~60th easiest)

The real test seems to be if we can solve 247ef758:2 and 16de56c4:2 using gpt-5.2-low. Another good test could be 88bcf3b4:2 which was not solved by the V6 solver, but 88bcf3b4:1 was solved by the V6 solver.

# Testing

I'm going to run 247ef758:2 and 16de56c4:2 repeatedly through gpt-5.2-low to see if it ever can solve it.

Result from 64 runs of each problem:
- 247ef758:2 solved 2/64 times
- 16de56c4:2 solved 0/64 times (but was solved once in another test run)

This is pretty extraordinary. These two are problems that the V6 solver struggled solving, and now is within the realm of solvable with gpt-5.2-low only.

Let's try 88bcf3b4:2 as well:
- 0/128 solved

Okay, well, it's capping out in its capability on gpt-5.2-low. Still, this is very impressive.
