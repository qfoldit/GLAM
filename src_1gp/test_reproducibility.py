"""Fast, dependency-light tests for deterministic GLAM search metadata."""
from reproducibility import make_search_rng, stable_config_id


def test_search_rng_repeats_for_same_seed():
    first = make_search_rng(2026)
    second = make_search_rng(2026)
    choices = [1, 2, 3, 4, 5]
    assert [first.choice(choices) for _ in range(20)] == [
        second.choice(choices) for _ in range(20)
    ]


def test_search_rng_changes_for_different_seed():
    first = make_search_rng(2026)
    second = make_search_rng(2027)
    choices = list(range(100))
    assert [first.choice(choices) for _ in range(20)] != [
        second.choice(choices) for _ in range(20)
    ]


def test_config_id_is_order_independent_and_sha256_length():
    first = {"seed": 7, "model": "graph", "hidden": 256}
    second = {"hidden": 256, "model": "graph", "seed": 7}
    assert stable_config_id(first) == stable_config_id(second)
    assert len(stable_config_id(first)) == 20
    assert stable_config_id(first) != stable_config_id({**first, "hidden": 512})
