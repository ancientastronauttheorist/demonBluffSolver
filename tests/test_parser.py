from demonbluff.io.parser import load_puzzle


def test_load_puzzle1():
    puzzle = load_puzzle("puzzle1.yaml")
    assert puzzle.seats == 8
    assert "alchemist" in puzzle.deck


def test_load_puzzle2():
    puzzle = load_puzzle("puzzle2.yaml")
    assert puzzle.seats == 8
    assert "baker" in puzzle.deck
