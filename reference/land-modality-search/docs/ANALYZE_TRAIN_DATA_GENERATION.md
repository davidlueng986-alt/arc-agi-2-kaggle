# Conclusion

Generated training data should only be used to assess confidence in the generated solver code. It should not be used to actually develop the solver.

In particular, when having multiple solvers, the solver that solves the most of the generated examples should be chosen as it is strictly (?) better.

# Strategies

When doing codegen the generalizability of the code is very important to measure, else we will have too high belief in code that solves all test examples and we will fail to see it not generalizing to test cases that may introduce new mechanics.

One way - but not the only way - to solve this is to generate more train input/output pairs.

A key thing to note when generating additional training data (input/output pairs) is that we uniquely DO NOT know the solution at the time we need to generate these cases. If we knew the solution we could easily deploy a wide variety of techniques. But because we do not know, only a few techniques remain:
- Color renaming
- Adding distractors
- Padding
- Reflection
- Rotation

# Assessment

Rotation/reflection fail on many problems. For example https://arcprize.org/play?task=31f7f899 which assumes a right-to-left sorting. A reflection of 180-degree rotation would brake this problem. Furthermore, a 90-degree rotation would introduce unnecessary complexity. Similarly on https://arcprize.org/play?task=7b5033c1 you may first thing that rotations would clear up simplistic assumption of top-to-bottom and distill them into the stricter "end-to-end", but in deriving the orientation of the output (vertical or horizontal) a 90-degree rotation on training input 2 actually makes this problem unsolvable (you can't determing output orientation).

Rotation can however be very helpful, like in https://arcprize.org/play?task=16de56c4 as an example. And, both rotation and reflection in this case works. In my tests with gpt-5.2-medium this problem is unsolvable, despite ~half of the time the solver gets all the training examples right. It fails to generalize to the right-to-left case - which would have been caught through generated training data.

Even color renaming, which seems harmless at first sight, brakes some problems. An example is https://arcprize.org/play?task=332f06d7 where renaming red and black makes the problem unsolvable.

I believe there are no completely safe ways to generate training data without knowing the solution transform. That said, generating training data can be helpful in building *some* confidence in the solver. If it also solves generated training data it is more likely to generalize, but even if failing the generated training data it still *can* be right.

# Can generated training data help determine accuracy if having multiple solvers?

Assume a case where you have multiple solvers that all solve the training data and generate the same answer to the test data. What will then the conclusion be if these solvers fail/succeed on different generated training data?

If they perform differently on the generated data it suggests that there exists a more generalizable solution, and only some of the solvers have figured this out. But, this more generalizable solution is not necessary to solve the problem.

Conclusion: No information obtainable from them performing differently

# If having multiple solvers generating different solutions, can generated data help in choosing the best one?

Yes, whichever solver solves the most of the generated data will should be chosen. Will it be more likely to be right? Or strickly be better? I think it'll strictly be better.


# TEST: Rotation, reflection, color

I added 9 augmented training examples for each given training example: 3 rotations, 3 reflections, and 3 color swaps (yes, one reflection overlapping with rotation)

Results below:
- 136b0064:1 : PASS (0.0/0.0/0.0) - e.g. all augmented examples failed, but solution was still right
- 16de56c4:1 : PASS (0.0/0.0/0.0)
- 247ef758:1 : PASS (0.0/0.33/0.0)
- 31f7f899:1 : FAIL (0.0/0.33/0.0) and PASS (0.0/0.33/1.0) - e.g. the color swap passing actually indicated it was a better solution, even though for this problem it doesn't seem to have any meaning
- 36a08778:1 : PASS (0.0/0.33/1.0)
- 7b5033c1:1 : FAIL (0.33/1.0/1.0)

It's questionable whether there is any signal here. Sorting of the solutions, at best.

Also when running it on a larger dataset with xlarge, the conclusion holds. There probably is not is no signal:
- 13e47133:2 : PASS (1.0/1.0/1.0) and FAIL (1.0/1.0/1.0)
- 3dc255db:1 : PASS (1.0/1.0/0.0), but FAIL (1.0/1.0/0.0) and FAIL (1.0/1.0/1.0)







