from harness.pool import SolutionPool


class TestSolutionPool:
    def test_dedup_identical_code(self):
        pool = SolutionPool(max_solutions=5)
        assert pool.add("def transform(g):\n    return g\n", "fb", 0.5)
        # whitespace-normalised duplicate
        assert not pool.add("def transform(g):\n    return  g\n", "fb", 0.7)
        assert len(pool) == 1

    def test_distinct_code_kept(self):
        pool = SolutionPool()
        pool.add("def transform(g):\n    return g\n", "a", 0.1)
        pool.add("def transform(g):\n    return g * 0\n", "b", 0.2)
        assert len(pool) == 2

    def test_feedback_block_top_k_by_score(self):
        pool = SolutionPool(max_solutions=2, improving_order=True)
        pool.add("code_low", "fb1", 0.2)
        pool.add("code_high", "fb2", 0.9)
        pool.add("code_mid", "fb3", 0.5)
        block = pool.feedback_block()
        # top-2 = 0.9, 0.5; improving order shows worst first
        assert "code_mid" in block and "code_high" in block
        assert "code_low" not in block
        assert block.index("0.50") < block.index("0.90")
        assert "<solution_1>" in block and "<solution_2>" in block

    def test_feedback_block_descending_when_not_improving(self):
        pool = SolutionPool(max_solutions=2, improving_order=False)
        pool.add("code_low", "fb1", 0.2)
        pool.add("code_high", "fb2", 0.9)
        block = pool.feedback_block()
        assert block.index("0.90") < block.index("0.20")

    def test_empty_block(self):
        pool = SolutionPool()
        assert pool.feedback_block() == ""
        assert pool.best() is None

    def test_best(self):
        pool = SolutionPool()
        pool.add("a", "fb", 0.3)
        pool.add("b", "fb", 0.8)
        assert pool.best()["code"] == "b"

    def test_scores_formatted_two_decimals(self):
        pool = SolutionPool(max_solutions=5)
        pool.add("a", "fb", 0.123456)
        assert "0.12" in pool.feedback_block()
