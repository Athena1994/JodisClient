

import multiprocessing
import sys


class StoppableInput:

    def __init__(self):
        self._input_process = None
        self._cancel = False
        self._mpm = multiprocessing.Manager()

    def read(self):
        def input_method(shared_dict):
            sys.stdin = open(0)
            shared_dict['result'] = input()

        shared_dict = self._mpm.dict()
        self._input_process = multiprocessing.Process(target=input_method,
                                                      args=(shared_dict, ))

        self._cancel = False

        self._input_process.start()
        self._input_process.join()

        return shared_dict.get('result')

    def cancel(self):
        if self._input_process is not None:
            self._input_process.terminate()
            self._cancel = True
            print()

    def was_cancelled(self):
        return self._cancel
