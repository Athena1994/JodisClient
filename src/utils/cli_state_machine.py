

from abc import abstractmethod
from dataclasses import dataclass
import logging
import multiprocessing
import sys
from typing import Callable, Dict, Generic, List, TypeVar, get_args


@dataclass
class Parameter:
    name: str
    default_value: object | None = None
    default_expr: str | None = None


class Handler:
    def __init__(self, callback: Callable[[dict, dict], 'BaseState'],
                 params: list[Parameter]):
        self._callback = callback
        self._params = params

    def handle(self, params: list, context: dict) -> 'BaseState':

        # named params are only allowed after positional params
        found_named_param = False
        abort = False
        positional_params = []
        named_params = {}
        for param in params:
            if '=' not in param:
                if found_named_param:
                    abort = True
                    break
                positional_params.append(param)
            else:
                found_named_param = True
                key, value = param.split('=')
                named_params[key] = value

        if abort:
            print('Named parameters must follow positional parameters!')
            return None

        parameter_dict = {}
        for i, param in enumerate(self._params):
            if i < len(positional_params):
                parameter_dict[param.name] = positional_params[i]
            else:
                if param.name in named_params:
                    parameter_dict[param.name] = named_params[param.name]
                else:
                    if param.default_value is not None:
                        parameter_dict[param.name] = param.default_value
                    elif param.default_expr is not None:
                        parameter_dict[param.name] = eval(param.default_expr)
                    else:
                        print(f'Missing parameter {param.name}')
                        return None

        return self._callback(parameter_dict, context)


class BaseState:
    def __init__(self):
        self._handlers: Dict[str, Handler] = {}

    def add_handler(self,
                    cmd: str,
                    callback: Callable[[dict, dict], 'BaseState'],
                    params: list[Parameter] = []):
        self._handlers[cmd] = Handler(callback, params)

    def handle(self, cmd: str, params: list, context: dict) -> 'BaseState':

        handler = self._handlers.get(cmd)
        if handler is None:
            return None

        return handler.handle(params, context)

    def prompt_prefix(self, context: dict) -> str:
        return ''


T = TypeVar("T")


class Command(Generic[T]):

    @abstractmethod
    def run(self, state: T) -> BaseState:
        pass

    def get_state_type(self) -> type:
        return T


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


class CLIStateMachine:

    def __init__(self):
        self._state = None

        self._command_queue: List[Command] = []
        self._input = StoppableInput()

    def dispatch_command(self, cmd: Command):
        self._command_queue.append(cmd)
        self._input.cancel()

    def run(self, context: dict, init_state: BaseState):
        self._state = init_state

        while True:
            while len(self._command_queue) != 0:
                cmd = self._command_queue.pop(0)
                if not isinstance(self._state,
                                  get_args(cmd.__orig_bases__[0])[0]):
                    logging.warning('Invalid state type for provided command')
                new_state = cmd.run(self._state)
                if new_state is not None:
                    self._state = new_state
            prompt = self._state.prompt_prefix(context)
            if len(prompt) != 0:
                print(f'{self._state.prompt_prefix(context)}: ', end='')

            cmd = self._input.read()

            if cmd is None:
                continue

            cmd_list = cmd.split()
            if len(cmd_list) == 0:
                continue

            command = cmd_list[0]
            params = cmd_list[1:]

            if command == 'exit':
                break

            if command == 'help':
                print('Available commands:')
                for cmd in self._state._handlers.keys():
                    print(f'\t{cmd}')
                continue

            result = self._state.handle(command, params, context)

            if result is None:
                print('Invalid command')
            else:
                self._state = result
