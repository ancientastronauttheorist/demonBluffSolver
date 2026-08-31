from dataclasses import fields
import tempfile
import unittest
from unittest.mock import patch

from game_loop import DecisionLog, GameSession, _release_session_lock, dispatch
from solver import CardInfo, DeckComposition, GameState


class TestGameStateIO(unittest.TestCase):
    def test_new_game_state_fields_are_appended_to_the_positional_abi(self):
        names = [field.name for field in fields(GameState)]
        self.assertEqual(
            names[-11:],
            [
                "executed_good_corrupted",
                "executed_good_roles",
                "board_count_provenance",
                "rambler_rule_version",
                "rambler_shut_up_observations",
                "baker_rule_version",
                "doppel_drunk_rule_version",
                "fortune_teller_rule_version",
                "terminal_loss_role",
                "executed_current_roles",
                "revealed_night_current_roles",
            ],
        )

    def test_round_trip_nested_deck(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Confessor"],
                outcasts=["Bombardier"],
                minions=["Minion"],
                demons=["Baa"],
            ),
            cards=[CardInfo(2, "Confessor", info_parsed={"dizzy": False})],
            n_evil=2,
            executed=[4],
            confirmed_evil=[4],
            executed_evil_roles={4: "Minion"},
            slayer_results=[{
                "slayer_pos": 2,
                "target_pos": 3,
                "killed": True,
                "revealed_role": "Wretch",
                "was_evil": False,
            }],
            hp=7,
            wrong_exec_cost=5,
            reveal_order=[2, 4],
            executed_good_corrupted={3: False},
            executed_good_roles={3: "Plague_Doctor"},
            terminal_loss_role="Bombardier",
            executed_current_roles={4: "Bombardier"},
            revealed_night_current_roles={3: "Witch"},
        )

        data = state.to_dict()
        loaded = GameState.from_dict(data)

        self.assertEqual(loaded, state)

    def test_loads_legacy_flat_deck_shape(self):
        data = {
            "n_cards": 4,
            "n_evil": 1,
            "villagers": ["Confessor", "Enlightened"],
            "outcasts": [],
            "minions": ["Minion"],
            "demons": [],
            "cards": [{"position": 1, "apparent_role": "Confessor", "info_parsed": {"dizzy": False}}],
            "executed_evil_roles": {"3": "Minion"},
        }

        loaded = GameState.from_dict(data)

        self.assertEqual(loaded.deck.villagers, ["Confessor", "Enlightened"])
        self.assertEqual(loaded.executed_evil_roles, {3: "Minion"})
        self.assertEqual(loaded.executed_good_roles, {})
        self.assertEqual(loaded.board_count_provenance, "legacy_unknown")
        self.assertIsNone(loaded.rambler_rule_version)
        self.assertEqual(loaded.rambler_shut_up_observations, [])
        self.assertIsNone(loaded.baker_rule_version)
        self.assertIsNone(loaded.doppel_drunk_rule_version)
        self.assertIsNone(loaded.fortune_teller_rule_version)
        self.assertIsNone(loaded.terminal_loss_role)
        self.assertEqual(loaded.executed_current_roles, {})
        self.assertEqual(loaded.revealed_night_current_roles, {})

    def test_game_session_save_load_round_trip_preserves_metadata(self):
        session = GameSession(5, 2)
        session.set_deck(
            villagers=["Confessor", "Bard"],
            outcasts=["Plague_Doctor"],
            minions=["Minion"],
            demons=["Pooka"],
        )
        session.board_villager_count = 2
        session.board_outcast_count = 1
        session.board_count_provenance = "trusted_pre_start"
        session.cards = [CardInfo(2, "Confessor", info_parsed={"dizzy": True})]
        session.used_abilities = [4]
        session.executed_good_corrupted = {1: False}
        session.executed_good_roles = {1: "Plague_Doctor"}
        session.terminal_loss_role = "Bombardier"
        session.executed_current_roles = {1: "Plague_Doctor"}
        session.revealed_night_current_roles = {5: "Witch"}
        session.slayer_results = [{
            "slayer_pos": 2,
            "target_pos": 5,
            "killed": True,
            "revealed_role": "Knight",
            "was_evil": True,
        }]
        session.rambler_shut_up_observations = [
            {"speaker_position": 2, "shut_up_target": 5},
        ]
        session.pd_ability_results = [{
            "pd_pos": 3,
            "target": 2,
            "is_corrupted": True,
            "evil_revealed": 5,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/session.json"
            session.save(path)
            loaded = GameSession.load(path)
            _release_session_lock()

        self.assertEqual(loaded.villagers, session.villagers)
        self.assertEqual(loaded.outcasts, session.outcasts)
        self.assertEqual(loaded.minions, session.minions)
        self.assertEqual(loaded.demons, session.demons)
        self.assertEqual(loaded.board_villager_count, 2)
        self.assertEqual(loaded.board_outcast_count, 1)
        self.assertEqual(loaded.board_count_provenance, "trusted_pre_start")
        self.assertEqual(loaded.rambler_rule_version, "rambler2_shut_up")
        self.assertEqual(loaded.baker_rule_version, "baker_day_reveal_v1")
        self.assertEqual(
            loaded.doppel_drunk_rule_version,
            "doppel_drunk_reveal_v1",
        )
        self.assertEqual(
            loaded.fortune_teller_rule_version,
            "fortune_teller_native_v1",
        )
        self.assertEqual(
            loaded.rambler_shut_up_observations,
            session.rambler_shut_up_observations,
        )
        self.assertEqual(loaded.used_abilities, [4])
        self.assertEqual(loaded.pd_ability_results, session.pd_ability_results)
        self.assertEqual(loaded.executed_good_corrupted, {1: False})
        self.assertEqual(loaded.executed_good_roles, {1: "Plague_Doctor"})
        self.assertEqual(loaded.terminal_loss_role, "Bombardier")
        self.assertEqual(loaded.executed_current_roles, {1: "Plague_Doctor"})
        self.assertEqual(loaded.revealed_night_current_roles, {5: "Witch"})
        self.assertEqual(loaded.slayer_results, session.slayer_results)

    def test_fresh_session_emits_current_rule_markers_but_legacy_state_does_not(self):
        current = GameSession(4, 1).to_game_state().to_dict()
        legacy = GameState.from_dict({
            "n_cards": 4,
            "n_evil": 1,
            "villagers": ["Lover"],
            "outcasts": ["Rambler"],
            "minions": ["Minion"],
            "demons": [],
        })

        self.assertEqual(current["rambler_rule_version"], "rambler2_shut_up")
        self.assertEqual(current["rambler_shut_up_observations"], [])
        self.assertEqual(current["baker_rule_version"], "baker_day_reveal_v1")
        self.assertEqual(
            current["doppel_drunk_rule_version"],
            "doppel_drunk_reveal_v1",
        )
        self.assertEqual(
            current["fortune_teller_rule_version"],
            "fortune_teller_native_v1",
        )
        self.assertIsNone(legacy.rambler_rule_version)
        self.assertEqual(legacy.rambler_shut_up_observations, [])
        self.assertIsNone(legacy.baker_rule_version)
        self.assertIsNone(legacy.doppel_drunk_rule_version)
        self.assertIsNone(legacy.fortune_teller_rule_version)
        self.assertIsNone(legacy.terminal_loss_role)
        self.assertNotIn("rambler_rule_version", legacy.to_dict())
        self.assertNotIn("baker_rule_version", legacy.to_dict())
        self.assertNotIn("doppel_drunk_rule_version", legacy.to_dict())
        self.assertNotIn("fortune_teller_rule_version", legacy.to_dict())
        self.assertNotIn("terminal_loss_role", legacy.to_dict())

    def test_reveal_order_defaults_empty_for_legacy_data(self):
        data = {
            "n_cards": 4,
            "n_evil": 1,
            "villagers": ["Confessor"],
            "outcasts": [],
            "minions": ["Minion"],
            "demons": [],
            "cards": [],
        }
        loaded = GameState.from_dict(data)
        self.assertEqual(loaded.reveal_order, [])

    def test_partial_deck_counts_do_not_promote_legacy_metadata(self):
        session = GameSession(5, 1)
        session.set_deck(["Confessor"], ["Wretch"], ["Minion"], ["Baa"])
        session.board_villager_count = 3
        session.board_outcast_count = 2
        session.board_count_provenance = "legacy_unknown"

        dispatch(
            "deck",
            ["V=Bard", "O=Bombardier", "M=Minion", "D=Pooka", "nv=3"],
            session,
        )

        self.assertEqual(session.villagers, ["Confessor"])
        self.assertEqual(session.board_villager_count, 3)
        self.assertEqual(session.board_outcast_count, 2)
        self.assertEqual(session.board_count_provenance, "legacy_unknown")

    def test_complete_deck_counts_are_saved_as_trusted(self):
        session = GameSession(5, 1)
        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_deck"),
        ):
            dispatch(
                "deck",
                ["V=Bard", "O=Bombardier", "M=Minion", "D=Pooka", "nv=3", "no=1"],
                session,
            )

        self.assertEqual(session.board_villager_count, 3)
        self.assertEqual(session.board_outcast_count, 1)
        self.assertEqual(session.board_count_provenance, "trusted_pre_start")

    def test_reveal_order_round_trip(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Confessor"],
                outcasts=[],
                minions=["Minion"],
                demons=["Baa"],
            ),
            cards=[CardInfo(3, "Confessor")],
            n_evil=1,
            reveal_order=[3, 1, 5],
        )
        data = state.to_dict()
        self.assertEqual(data["reveal_order"], [3, 1, 5])
        loaded = GameState.from_dict(data)
        self.assertEqual(loaded.reveal_order, [3, 1, 5])

    def test_session_add_card_populates_reveal_order(self):
        session = GameSession(5, 1)
        session.set_deck(["Confessor", "Bard"], [], ["Minion"], ["Baa"])
        session.add_card(CardInfo(3, "Confessor"))
        session.add_card(CardInfo(1, "Bard"))
        session.add_card(CardInfo(3, "Confessor"))  # re-read, should not duplicate
        self.assertEqual(session.reveal_order, [3, 1])
        self.assertIsNone(session.baker_rule_version)

    def test_verified_card_entry_preserves_current_baker_marker(self):
        session = GameSession(5, 1)
        session.reveal_order = [3]

        session.add_card(CardInfo(3, "Confessor"))

        self.assertEqual(session.reveal_order, [3])
        self.assertEqual(session.baker_rule_version, "baker_day_reveal_v1")


if __name__ == "__main__":
    unittest.main()
