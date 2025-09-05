from demonbluff.core.model import Alignment, Status, Assignment
from demonbluff.core.world import World


def test_assignment_hash_and_eq():
    a1 = Assignment("fortune teller", Alignment.GOOD, frozenset({Status.POISONED}))
    a2 = Assignment("fortune teller", Alignment.GOOD, frozenset({Status.POISONED}))
    assert a1 == a2
    assert hash(a1) == hash(a2)


def test_world_hash_and_access():
    world1 = World({1: Assignment("knitter", Alignment.GOOD)})
    world2 = World({1: Assignment("knitter", Alignment.GOOD)})
    assert world1 == world2
    assert hash(world1) == hash(world2)
    assert world1.seat(1).role == "knitter"
