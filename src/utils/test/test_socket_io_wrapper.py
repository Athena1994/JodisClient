import logging
from multiprocessing import Process
import time
import unittest
import unittest.async_case

from flask import Flask
from flask_socketio import SocketIO
import flask_socketio

from src.utils.socket_io_wrapper import SocketIOWrapper


logging.basicConfig(level=logging.DEBUG)
PORT = 12345


def server():
    class TestNamespace(flask_socketio.Namespace):
        def __init__(self, namespace):
            super().__init__(namespace)

        def on_call(self, event):
            print(event)
            if event == 'a':
                self.emit('test_a', {'test': 42})
            elif event == 'b':
                self.emit('test_b', {'test': 43})

        def on_ok(self, t):
            self.emit(t, 'ok')

    app = Flask('testapp')
    socketio = SocketIO(app)
    socketio.on_namespace(TestNamespace('/test'))
    socketio.run(app, use_reloader=True, debug=True,
                 host='localhost', port=PORT,
                 allow_unsafe_werkzeug=True)


class TestSocketIOWrapper(unittest.TestCase):

    def test_socket_io_wrapper(self):
        print("start\n\n")

        server_p = Process(target=server, name='server')
        server_p.start()

        socket = SocketIOWrapper(2)

        def test_a_handler(msg):
            self.assertEqual(msg, {'test': 42})
            socket.emit('ok', 'a', namespace='/test')

        def test_b_handler(msg):
            self.assertEqual(msg, {'test': 43})
            socket.emit('ok', 'b', namespace='/test')

        socket.register_handler('test_a', test_a_handler)
        socket.register_handler('test_b', test_b_handler)

        socket.connect('http://localhost', PORT, '/test')
        self.assertTrue(socket.is_connected())

        socket.emit('call', 'a', namespace='/test')
        resp = socket.receive()
        self.assertEqual(resp.event, 'a')
        self.assertEqual(resp.data, 'ok')

        socket.emit('call', 'b', namespace='/test')
        resp = socket.receive()
        self.assertEqual(resp.event, 'b')
        self.assertEqual(resp.data, 'ok')

        t = time.time_ns()
        with self.assertRaises(TimeoutError):
            socket.receive()
        self.assertAlmostEqual(time.time_ns() - t, 2 * 1e9, delta=1e8)
        print('done')
        server_p.terminate()


if __name__ == '__main__':
    unittest.main()
