

"""

    Evaluation commands are used to evaluate actions.
    On activation commands will be provided with the current environment 
    state, and sample specific information (e.g. current candle) which can 
    be accessed by writing "{var_name}" in the command string.   

    Context Data:
        A program may provide context data with accessible to the command
        variables to the execution command of a processor.

    Value access:
        stored variables can be accessed by setting the variable name in 
        curved brackets. E.g. "{var_name}". If a variable is not found, 
        the expresion will evaluate to "False".

    Constants:
        Constants are stored in the 'constants' key.

        example:
            "constants": { 
                "buy_price": 100
                "sell_price": 200
            }

    Commands: 
        Commands are stored in the 'process' key and are executed in the 
        order they are provided. 

        example:
           "process": [
                "program that temporarily stores test and immediately returns 
                its value",
                {"!": "set", "name": "test", value: 100}
                {"!": "return", "value": "{test}"}
            ]

        Environment commands:
        - store: 
            description: "Store a variable in the environment state that
            will be retained even after program ends. A possible previous 
            value will be overwritten."
            
            params: 
                "name": str  -> variable name to be stored in environment 
                                state
                "value": any -> value to be stored in environment state
                                if this parameter is omitted, stored value 
                                will be 'True'

        - clear:
            description: "Removes variable from environment state."
            params: 
                "name": str -> variable name to be removed from environment 
                               state

        - set:
            description: "Set temporary value which will be removed after 
                          returning reward"
            params: 
                "name": str -> variable name to be set in environment state
                "value": any -> value to be set in environment state

        Control commands:
        - if:
            description: "Execute command if condition is met."
            params:
                "condition": str -> condition to be checked
                "then": list -> commands if condition is met   
                "else": list -> commands if condition is not met  

        - switch:
            description: "Execute command based on condition."
            params:
                "condition": str -> condition to be checked
                "cases": list -> list of cases
                    "value": str -> condition to be checked
                    "then": list -> commands if condition is met

        - "return": 
            description: "Set reward for the current sample and stop 
                          execution."
            params:
                   "value": float -> reward value
"""



from abc import abstractmethod
import copy
import re
from typing import List


class Expression:
    def __init__(self, expression: str):
        self._expression = expression

        regex = r"{(.*?)}"
        matches = re.findall(regex, self._expression)
        for match in matches:
            self._expression = self._expression.replace(f"{{{match}}}", 
                                                        f"env['{match}']")

    def evaluate(self, env: dict):
        try:
            return eval(self._expression)        
        except KeyError as e:
            raise ValueError(f"Variable {e} not found in environment")

class Command:
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, env: dict) -> None:
        pass

class CommandFactory:
    @staticmethod
    def create_command(desc: dict) -> Command:
        if "!" not in desc:
            raise ValueError("Command type not found")
        
        if desc['!'] == "store":
            return StoreCommand.from_config(desc)
        elif desc['!'] == "clear":
            return ClearCommand.from_config(desc)
        elif desc['!'] == "set":
            return SetCommand.from_config(desc)
        elif desc['!'] == "if":
            return IfCommand.from_config(desc)
        elif desc['!'] == "switch":
            return SwitchCommand.from_config(desc)
        elif desc['!'] == "return":
            return ReturnCommand.from_config(desc)
        else:
            raise ValueError("Command type not supported")

class CommandProcessor:
    def __init__(self, program: dict):

        self._working_env = {}

        if "constants" in program:
            self._constants = copy.deepcopy(program["constants"])
        else:
            self._constants = {}

        if "process" not in program:
            raise ValueError("No commands found! (process key missing)")
        
        self._command_sequence \
            = CommandSequence.from_list(program["process"])

        self.reset()
        
    def reset(self):
        self._working_env = copy.deepcopy(self._constants)

    def execute(self, context: dict):
        env = {'values': self._working_env, 
               'local_vars': set(context.keys())}
        env['values'].update(context)

        self._command_sequence.execute(env)
            
        # remove local variables
        for key in env['local_vars']:
            env['values'].pop(key, None)

        return env.get("return", None)
        
class CommandSequence(Command):
    def __init__(self, commands: List[Command]):
        super().__init__()
        self._commands = commands

    def execute(self, env: dict) -> None:
        for command in self._commands:
            command.execute(env)
            if "return" in env:
                break            

    @staticmethod
    def from_list(config: List[dict]):
        commands = [CommandFactory.create_command(command) 
                    for command in config]
        return CommandSequence(commands)

class StoreCommand(Command):
    def __init__(self, name: str, value: str):
        super().__init__()
        self._var_name = name
        self._value_expr = Expression(value)

    def execute(self, env: dict):
        env['values'][self._var_name] \
              = self._value_expr.evaluate(env['values'])

    @staticmethod
    def from_config(config: dict):
        if "name" not in config:
            raise ValueError("Variable name not found")
        if "value" not in config:
            config["value"] = True
        return StoreCommand(config["name"], config["value"])
    

class ClearCommand(Command):
    def __init__(self, name: str):
        super().__init__()
        self._var_name = name

    def execute(self, env: dict):
        if self._var_name in env['local_vars']:
            env['local_vars'].remove(self._var_name)
        env['values'].pop(self._var_name, None)

    @staticmethod
    def from_config(config: dict):
        if "name" not in config:
            raise ValueError("Variable name not found")
        return ClearCommand(config["name"])
    
class SetCommand(Command):
    def __init__(self, name: str, value: str):
        super().__init__()
        self._var_name = name
        self._value_expr = Expression(value)

    def execute(self, env: dict):
        env['values'][self._var_name] \
              = self._value_expr.evaluate(env['values'])
        env['local_vars'].add(self._var_name)

    @staticmethod
    def from_config(config: dict):
        if "name" not in config:
            raise ValueError("Variable name not found")
        if "value" not in config:
            raise ValueError("Value not found")
        return SetCommand(config["name"], config["value"])
    
class IfCommand(Command):
    def __init__(self, 
                 condition: str, 
                 then: CommandSequence, 
                 else_: CommandSequence):
        super().__init__()
        self._condition = Expression(condition)
        self._then = then
        self._else = else_

    def execute(self, env: dict):
        if self._condition.evaluate(env['values']):
            self._then.execute(env)
        elif self._else is not None:
                self._else.execute(env)

    @staticmethod
    def from_config(config: dict):
        if "condition" not in config:
            raise ValueError("Condition not found")
        if "then" not in config:
            raise ValueError("Then not found")
        if "else" not in config:
            else_ = None
        else:
            else_ = CommandSequence.from_list(config["else"])
        return IfCommand(config["condition"], 
                         CommandSequence.from_list(config["then"]),
                         else_)
    
class SwitchCommand(Command):
    def __init__(self, 
                 condition: str, 
                 cases: List[dict]):
        super().__init__()
        self._condition = Expression(condition)
        self._cases = {}
        for case in cases:
            if "value" not in case:
                raise ValueError("Value not found")
            if "then" not in case:
                raise ValueError("Then not found")
            self._cases[case["value"]] = CommandSequence.from_list(case["then"])
            
    def execute(self, env: dict):
        value = self._condition.evaluate(env['values'])

        seq = self._cases.get(value, None)
        if seq is None:
            raise ValueError(f"Value {value} not expected in switch.")

        seq.execute(env)


    @staticmethod
    def from_config(config: dict):
        if "condition" not in config:
            raise ValueError("Condition not found")
        if "cases" not in config:
            raise ValueError("Cases not found")
        return SwitchCommand(config["condition"], config["cases"])
    

class ReturnCommand(Command):
    def __init__(self, value: str):
        super().__init__()
        self._value_expr = Expression(value)

    def execute(self, env: dict):
        env['return'] = self._value_expr.evaluate(env['values'])

    @staticmethod
    def from_config(config: dict):
        if "value" not in config:
            raise ValueError("Value not found")
        return ReturnCommand(config["value"])