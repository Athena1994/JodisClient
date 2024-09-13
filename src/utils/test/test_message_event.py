
import threading
import time
import unittest
import unittest.async_case

from utils.message_event import MessageEvent


class TestMessageEvent(unittest.TestCase):

    def test_message_event(self):
        event = MessageEvent()

        # assert timeout is raised
        t = time.time_ns()

        with self.assertRaises(TimeoutError):
            event.wait(0.5)

        self.assertAlmostEqual(time.time_ns() - t, 0.5 * 1e9, delta=1e8)

        # assert message is received
        def delayed_set():
            time.sleep(1)
            event.set(42)

        threading.Thread(target=delayed_set).start()

        self.assertNotEqual(event.get_message(), 42)
        self.assertEqual(event.wait(2), 42)
        self.assertEqual(event.get_message(), 42)
