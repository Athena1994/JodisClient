

import unittest

from core.cli_state_machine.state_machine_control import StateMachineControl


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.state_machine = StateMachineControl()

    def test_initial_state(self):
        self.assertEqual(self.state_machine.current_state, "initial")

    def test_transition_to_next_state(self):
        self.state_machine.transition_to("next")
        self.assertEqual(self.state_machine.current_state, "next")

    def test_invalid_transition(self):
        with self.assertRaises(ValueError):
            self.state_machine.transition_to("invalid")
