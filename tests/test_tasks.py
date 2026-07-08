"""
test_tasks.py

Unit tests for task generators in attn_phase.tasks.

Verifies that every task generator produces well-formed output, that
generate_task_set produces the right number of independently-seeded
instances, and that the required dict keys are always present.
"""

import pytest
from attn_phase.tasks import (
    make_copy_task,
    make_easy_modular_arithmetic_task,
    make_lookup_task,
    make_sorting_task,
    make_modular_arithmetic_task,
    generate_task_set,
    answer_matches,
    extract_model_answer,
)

REQUIRED_KEYS = {"task_id", "task_type", "prompt", "expected_answer", "n_examples"}


# ---------------------------------------------------------------------------
# Required keys present in every task
# ---------------------------------------------------------------------------

class TestRequiredKeys:

    def test_copy_task_has_required_keys(self):
        task = make_copy_task(seed=42)
        assert REQUIRED_KEYS.issubset(task.keys())

    def test_easy_mod_arith_has_required_keys(self):
        task = make_easy_modular_arithmetic_task(seed=42)
        assert REQUIRED_KEYS.issubset(task.keys())

    def test_lookup_has_required_keys(self):
        task = make_lookup_task(seed=42)
        assert REQUIRED_KEYS.issubset(task.keys())

    def test_sorting_has_required_keys(self):
        task = make_sorting_task(seed=42)
        assert REQUIRED_KEYS.issubset(task.keys())

    def test_mod_arith_has_required_keys(self):
        task = make_modular_arithmetic_task(
            seed=42, modulus=97, n_digits=2, n_examples=10)
        assert REQUIRED_KEYS.issubset(task.keys())


# ---------------------------------------------------------------------------
# Prompt and expected_answer are non-empty strings
# ---------------------------------------------------------------------------

class TestPromptContent:

    def test_copy_prompt_is_nonempty(self):
        task = make_copy_task(seed=42, n_examples=10)
        assert isinstance(task["prompt"], str) and len(task["prompt"]) > 0
        assert isinstance(task["expected_answer"], str) and len(task["expected_answer"]) > 0

    def test_lookup_expected_answer_matches_prompt_pattern(self):
        task = make_lookup_task(seed=42, n_keys=4, n_examples=10)
        # Expected answer should be one of the values in the mapping
        assert task["expected_answer"].startswith("v")

    def test_sorting_expected_answer_is_sorted(self):
        task = make_sorting_task(seed=42, n_numbers=4, n_examples=5)
        expected_nums = list(map(int, task["expected_answer"].split()))
        assert expected_nums == sorted(expected_nums)

    def test_mod_arith_expected_answer_is_correct(self):
        # Verify the expected answer is actually correct modular arithmetic
        task = make_modular_arithmetic_task(
            seed=42, modulus=10, n_digits=1, n_examples=5)
        # Extract the final query from the prompt
        prompt = task["prompt"]
        # Last part is "a + b mod m ="
        last_expr = prompt.split(" , ")[-1].strip()
        # Parse: "a + b mod m ="
        parts = last_expr.split()
        a, op, b = int(parts[0]), parts[1], int(parts[2])
        m = int(parts[4])
        expected = (a + b) % m if op == "+" else (a * b) % m
        assert int(task["expected_answer"]) == expected


# ---------------------------------------------------------------------------
# generate_task_set
# ---------------------------------------------------------------------------

class TestGenerateTaskSet:

    def test_correct_number_of_instances(self):
        tasks = generate_task_set("lookup", n_instances=5, base_seed=42,
                                   n_keys=8, n_examples=20)
        assert len(tasks) == 5

    def test_all_instances_have_required_keys(self):
        tasks = generate_task_set("sorting", n_instances=3, base_seed=42,
                                   n_numbers=4, n_examples=10)
        for t in tasks:
            assert REQUIRED_KEYS.issubset(t.keys())

    def test_instances_have_unique_task_ids(self):
        tasks = generate_task_set("lookup", n_instances=5, base_seed=42,
                                   n_keys=8, n_examples=20)
        ids = [t["task_id"] for t in tasks]
        assert len(set(ids)) == 5   # all unique

    def test_instances_have_different_prompts(self):
        # Different seeds should produce genuinely different prompts
        tasks = generate_task_set("easy_mod_arith", n_instances=3,
                                   base_seed=42, n_examples=10)
        prompts = [t["prompt"] for t in tasks]
        assert len(set(prompts)) == 3

    def test_reproducible_from_same_base_seed(self):
        tasks_a = generate_task_set("lookup", n_instances=3, base_seed=99,
                                     n_keys=4, n_examples=10)
        tasks_b = generate_task_set("lookup", n_instances=3, base_seed=99,
                                     n_keys=4, n_examples=10)
        for a, b in zip(tasks_a, tasks_b):
            assert a["prompt"] == b["prompt"]
            assert a["expected_answer"] == b["expected_answer"]

    def test_different_base_seeds_give_different_tasks(self):
        tasks_a = generate_task_set("lookup", n_instances=1, base_seed=1,
                                     n_keys=4, n_examples=10)
        tasks_b = generate_task_set("lookup", n_instances=1, base_seed=2,
                                     n_keys=4, n_examples=10)
        assert tasks_a[0]["prompt"] != tasks_b[0]["prompt"]

    def test_instance_idx_is_set(self):
        tasks = generate_task_set("sorting", n_instances=4, base_seed=42,
                                   n_numbers=3, n_examples=5)
        for i, t in enumerate(tasks):
            assert t["instance_idx"] == i

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            generate_task_set("nonexistent_task", n_instances=3, base_seed=42)