"""
tasks.py — Synthetic task generation for the Attention Phase Analyzer.

Provides five task types used to probe within-context attention dynamics
in causal language models:

    copy          : repeat a token sequence, query on a new token
    transduction  : learn a fixed token-to-token mapping, query on a key
    mod_arith     : modular arithmetic few-shot (a + b mod m = ?)
    lookup        : key=value store, query on a previously-seen key
    sorting       : sort N small integers ascending, few-shot then query

Each task function returns a dict with these guaranteed keys:
    task_id         : str  — unique identifier
    task_type       : str  — one of the five types above
    prompt          : str  — the full few-shot prompt fed to the model
    expected_answer : str  — the correct completion (may be multi-token)
    n_examples      : int  — number of few-shot examples in the prompt

Length-matched variants (copy, transduction) grow n_examples until the
tokenized prompt lands within a tolerance band of a target token count.
This is used to control for sequence length as a confound.

The difficulty-sweep helper (`sweep_difficulty_for_failure`) finds a
mod_arith variant that GPT-2 small FAILS while matching a target length —
used to get a hard-but-failed comparison task.

`generate_task_set` produces multiple independently-seeded instances of
any task type for statistical testing (Phase C1).
"""

import random
import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)


def _derive_seed(base_seed: int, instance_idx: int) -> int:
    """Deterministic per-instance seed derived from a base seed."""
    return base_seed * 1000 + instance_idx


def count_tokens(text: str, tokenizer) -> int:
    """Exact token count using the model's own tokenizer."""
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def make_copy_task(seed: int, n_examples: int = 40, vocab=None) -> dict:
    """
    Copy task: show pairs of identical tokens, then query on a new token.
    The model must output that token unchanged.
    GPT-2 small reliably fails this when the vocabulary is large enough
    that the query token is rare in the few-shot context.
    """
    _set_seed(seed)
    if vocab is None:
        vocab = [f"tok{i}" for i in range(30)]
    pairs = [f"{random.choice(vocab)} {random.choice(vocab)}" for _ in range(n_examples)]
    # Make each pair actually a copy pair
    pairs = []
    for _ in range(n_examples):
        t = random.choice(vocab)
        pairs.append(f"{t} {t}")
    body = " , ".join(pairs)
    query_tok = random.choice(vocab)
    prompt = f"{body} , {query_tok}"
    return {
        "task_id": f"copy_seed{seed}",
        "task_type": "copy",
        "prompt": prompt,
        "expected_answer": query_tok,
        "n_examples": n_examples,
    }


def make_copy_task_matched(seed: int, tokenizer, target_tokens: int,
                            tol: float = 0.05, vocab=None,
                            max_examples: int = 200) -> dict:
    """
    Length-matched copy task. Grows n_examples until the tokenized prompt
    is within `tol` fraction of `target_tokens`. Used to eliminate
    sequence length as a confound when comparing against a longer task.
    """
    _set_seed(seed)
    if vocab is None:
        vocab = [f"tok{i}" for i in range(30)]

    def build(n):
        _set_seed(seed)          # re-seed so vocab choices are reproducible
        pairs = []
        for _ in range(n):
            t = random.choice(vocab)
            pairs.append(f"{t} {t}")
        body = " , ".join(pairs)
        q = random.choice(vocab)
        return f"{body} , {q}", q

    n_examples = 10
    prompt, expected = build(n_examples)
    n_tok = count_tokens(prompt, tokenizer)
    lower, upper = target_tokens * (1 - tol), target_tokens * (1 + tol)

    while n_tok < lower and n_examples < max_examples:
        n_examples += 2
        prompt, expected = build(n_examples)
        n_tok = count_tokens(prompt, tokenizer)

    return {
        "task_id": f"copy_matched_{target_tokens}_seed{seed}",
        "task_type": "copy_matched",
        "prompt": prompt,
        "expected_answer": expected,
        "n_examples": n_examples,
        "actual_tokens": n_tok,
        "target_tokens": target_tokens,
    }


def make_transduction_task_matched(seed: int, tokenizer, target_tokens: int,
                                    tol: float = 0.05,
                                    max_examples: int = 200) -> dict:
    """
    Length-matched transduction task. Maps each letter in a small alphabet
    to a fixed arbitrary other letter (e.g. a->q, b->z), shows repeated
    examples, then queries an unseen instance. Length-matched to control
    for sequence length confound.
    """
    _set_seed(seed)
    alphabet = list("abcdefghij")
    shuffled = alphabet[:]
    random.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))

    def build(n):
        _set_seed(seed)
        examples = []
        for _ in range(n):
            k = random.choice(alphabet)
            examples.append(f"{k} -> {mapping[k]}")
        body = " , ".join(examples)
        q = random.choice(alphabet)
        return f"{body} , {q} ->", mapping[q]

    n_examples = 10
    prompt, expected = build(n_examples)
    n_tok = count_tokens(prompt, tokenizer)
    lower, upper = target_tokens * (1 - tol), target_tokens * (1 + tol)

    while n_tok < lower and n_examples < max_examples:
        n_examples += 2
        prompt, expected = build(n_examples)
        n_tok = count_tokens(prompt, tokenizer)

    return {
        "task_id": f"transduction_matched_{target_tokens}_seed{seed}",
        "task_type": "transduction_matched",
        "prompt": prompt,
        "expected_answer": expected,
        "n_examples": n_examples,
        "actual_tokens": n_tok,
        "target_tokens": target_tokens,
    }


def make_modular_arithmetic_task(seed: int, modulus: int, n_digits: int,
                                  n_examples: int, op: str = "+") -> dict:
    """
    Modular arithmetic few-shot prompt:
        "a + b mod m = c , a + b mod m = c , ... , a + b mod m ="
    with a,b drawn to have `n_digits` digits. The model is queried on
    a final unseen pair.

    Difficulty is controlled by modulus and digit count:
    larger modulus + more digits = harder for GPT-2 small to track.
    """
    _set_seed(seed)
    low = 10 ** (n_digits - 1) if n_digits > 1 else 0
    high = 10 ** n_digits - 1

    def sample_pair():
        a = random.randint(low, high)
        b = random.randint(low, high)
        c = (a + b) % modulus if op == "+" else (a * b) % modulus
        return a, b, c

    examples = []
    for _ in range(n_examples):
        a, b, c = sample_pair()
        examples.append(f"{a} {op} {b} mod {modulus} = {c}")
    body = " , ".join(examples)
    qa, qb, qc = sample_pair()
    prompt = f"{body} , {qa} {op} {qb} mod {modulus} ="

    return {
        "task_id": f"mod_arith_m{modulus}_d{n_digits}_seed{seed}",
        "task_type": "mod_arith",
        "prompt": prompt,
        "expected_answer": str(qc),
        "n_examples": n_examples,
        "modulus": modulus,
        "n_digits": n_digits,
    }


def make_easy_modular_arithmetic_task(seed: int, n_examples: int = 20) -> dict:
    """Small modulus, single digit — GPT-2 small can sometimes solve this."""
    return make_modular_arithmetic_task(
        seed=seed, modulus=10, n_digits=1, n_examples=n_examples, op="+",
    )


def make_lookup_task(seed: int, n_keys: int = 8, n_examples: int = 20) -> dict:
    """
    Key=value lookup. Shows repeated key=value pairs, then queries a
    previously-seen key. Easier than transduction since the query key is
    guaranteed to have appeared in the few-shot block.
    """
    _set_seed(seed)
    keys = [f"k{i}" for i in range(n_keys)]
    values = [f"v{i}" for i in range(n_keys)]
    mapping = dict(zip(keys, values))

    shown = [f"{k} = {mapping[k]}" for k in
             (random.choice(keys) for _ in range(n_examples))]
    body = " , ".join(shown)
    query_k = random.choice(keys)
    prompt = f"{body} , {query_k} ="

    return {
        "task_id": f"lookup_seed{seed}",
        "task_type": "lookup",
        "prompt": prompt,
        "expected_answer": mapping[query_k],
        "n_examples": n_examples,
    }


def make_sorting_task(seed: int, n_numbers: int = 4,
                       n_examples: int = 15) -> dict:
    """Sort n_numbers small integers ascending; few-shot then query."""
    _set_seed(seed)

    def build_example():
        nums = [random.randint(0, 9) for _ in range(n_numbers)]
        return (f"sort {' '.join(map(str, nums))} -> "
                f"{' '.join(map(str, sorted(nums)))}")

    body = " , ".join(build_example() for _ in range(n_examples))
    query_nums = [random.randint(0, 9) for _ in range(n_numbers)]
    prompt = f"{body} , sort {' '.join(map(str, query_nums))} ->"
    expected = " ".join(map(str, sorted(query_nums)))

    return {
        "task_id": f"sorting_seed{seed}",
        "task_type": "sorting",
        "prompt": prompt,
        "expected_answer": expected,
        "n_examples": n_examples,
    }


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------

def extract_model_answer(generated_text: str, prompt: str) -> str:
    """
    Returns the model's raw continuation after the prompt, stripped of
    leading/trailing whitespace.

    Does NOT extract or parse anything — shape-aware comparison is handled
    by answer_matches() below. Earlier versions regex-matched only the first
    integer, which silently mis-scored every non-arithmetic task (lookup's
    "v0" became "0"; sorting's "1 3 4 6" became "1"). This version returns
    the full continuation and lets answer_matches() do the comparison.
    """
    return generated_text[len(prompt):].strip()


def answer_matches(model_continuation: str,
                   expected_answer: str) -> tuple[bool, str]:
    """
    Shape-aware comparison between the model's raw continuation and the
    task's expected answer. Handles three cases:

    1. Multi-token expected (sorting: "1 3 4 6"):
       Take the first N whitespace-delimited tokens from the continuation
       where N = number of tokens in expected_answer, compare the sequence.

    2. Single numeric token expected (mod_arith: "53"):
       Extract the first integer from the continuation and compare.
       Handles cases where the model continues generating past the answer.

    3. Single non-numeric token expected (lookup: "v0", copy: "tok22"):
       Take the first whitespace-delimited token and compare directly.
       Earlier regex-only approach broke here: extracting "0" from "v0"
       produced a spurious mismatch even when the answer was correct.

    Returns (solved: bool, extracted: str) where `extracted` is what was
    actually compared, for logging and debugging.
    """
    continuation = model_continuation.strip()
    expected = expected_answer.strip()
    expected_tokens = expected.split()

    if len(expected_tokens) > 1:
        got_tokens = continuation.split()[:len(expected_tokens)]
        extracted = " ".join(got_tokens)
        return extracted == expected, extracted

    if expected.lstrip("-").isdigit():
        match = re.search(r"-?\d+", continuation)
        extracted = match.group(0) if match else None
        return (extracted == expected), extracted

    got_token = continuation.split()[0] if continuation.split() else None
    return (got_token == expected), got_token


# ---------------------------------------------------------------------------
# Difficulty sweep — finds a hard-but-failed, length-matched mod_arith task
# ---------------------------------------------------------------------------

def sweep_difficulty_for_failure(
        tokenizer, model, generate_fn, seed: int, target_tokens: int,
        tol: float = 0.08,
        modulus_candidates: tuple = (97, 131, 211, 503, 997),
        digit_candidates: tuple = (2, 3, 4),
        n_examples_range: tuple = (10, 40),
) -> tuple[dict | None, list]:
    """
    Sweeps (modulus, n_digits, n_examples) to find a modular arithmetic
    variant that:
      (a) GPT-2 small gets WRONG (the failure condition we need), AND
      (b) lands within `tol` of target_tokens (length-matched)

    Used to construct a hard-but-failed comparison task that breaks the
    original confound where solvedness, difficulty, and length were all
    perfectly correlated (n=1 on the solved side).

    Returns (task_dict | None, sweep_log). task_dict is None if no config
    in the grid satisfies both conditions — widen the grid and re-run.
    """
    log = []
    lower, upper = target_tokens * (1 - tol), target_tokens * (1 + tol)

    for modulus in modulus_candidates:
        for n_digits in digit_candidates:
            for n_examples in range(n_examples_range[0],
                                    n_examples_range[1] + 1, 2):
                task = make_modular_arithmetic_task(
                    seed=seed, modulus=modulus, n_digits=n_digits,
                    n_examples=n_examples, op="+",
                )
                n_tok = count_tokens(task["prompt"], tokenizer)
                task["actual_tokens"] = n_tok

                if n_tok < lower:
                    continue
                if n_tok > upper:
                    break

                generated = generate_fn(task["prompt"], max_new_tokens=10)
                continuation = extract_model_answer(generated, task["prompt"])
                solved, model_answer = answer_matches(
                    continuation, task["expected_answer"])

                log.append({
                    "modulus": modulus, "n_digits": n_digits,
                    "n_examples": n_examples, "tokens": n_tok,
                    "solved": bool(solved), "model_answer": model_answer,
                    "expected": task["expected_answer"],
                })

                if not solved:
                    task["solved"] = False
                    task["model_answer"] = model_answer
                    return task, log

    return None, log


# ---------------------------------------------------------------------------
# Multi-seed task set generation (required for Phase C1 statistical testing)
# ---------------------------------------------------------------------------

TASK_BUILDERS = {
    "copy":          lambda seed, **kw: make_copy_task(seed, **kw),
    "lookup":        lambda seed, **kw: make_lookup_task(seed, **kw),
    "sorting":       lambda seed, **kw: make_sorting_task(seed, **kw),
    "easy_mod_arith": lambda seed, **kw: make_easy_modular_arithmetic_task(
        seed, **kw),
    "mod_arith":     lambda seed, **kw: make_modular_arithmetic_task(
        seed, **kw),
}


def generate_task_set(task_type: str, n_instances: int,
                       base_seed: int, **task_kwargs) -> list[dict]:
    """
    Generates `n_instances` independently-seeded task dicts of the given
    task_type. Seeds are derived deterministically from base_seed so the
    full set is reproducible from a single number.

        seed_i = base_seed * 1000 + i   (i = 0 .. n_instances-1)

    This means:
    - Same base_seed always produces the same task set
    - Different base_seeds produce genuinely different task sets
    - Instances within a set are independent of each other (different seeds)

    task_type must be one of the keys in TASK_BUILDERS.
    task_kwargs are passed through to the underlying task generator
    (e.g. modulus=97, n_digits=2 for mod_arith).

    Returns a list of task dicts, each with a unique task_id that encodes
    the instance index so results can be traced back to their seed.

    Example:
        tasks = generate_task_set("lookup", n_instances=5, base_seed=42,
                                   n_keys=8, n_examples=20)
        # produces 5 lookup tasks with seeds 42000, 42001, 42002, 42003, 42004
    """
    if task_type not in TASK_BUILDERS:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Choose from: {list(TASK_BUILDERS.keys())}"
        )

    builder = TASK_BUILDERS[task_type]
    tasks = []
    for i in range(n_instances):
        seed_i = _derive_seed(base_seed, i)
        task = builder(seed_i, **task_kwargs)
        # Append instance index to task_id so it's unique across the set
        task["task_id"] = f"{task['task_id']}_inst{i}"
        task["instance_idx"] = i
        task["base_seed"] = base_seed
        tasks.append(task)
    return tasks