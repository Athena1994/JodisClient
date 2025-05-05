from dataclasses import dataclass
from typing import List


@dataclass
class CLICommand:
    name: str
    positional_params: List[object]
    named_params: dict[str, object]

    @property
    def positional_parameter_count(self) -> int:
        return len(self.positional_params)

    def get_parameter(self, ix: int, name: str) -> object:
        if ix < self.positional_parameter_count:
            if name in self.named_params:
                raise ValueError(
                    f"Positional parameter {ix+1} and named parameter {name} "
                    "cannot be used together")
            return self.positional_params[ix]

        return self.named_params.get(name, None)

    @staticmethod
    def parse(cmd: str) -> 'CLICommand':
        cmd_list = cmd.split()
        if len(cmd_list) == 0:
            return None

        command = cmd_list[0]
        params = cmd_list[1:]

        # parse parameters
        raw_params = [CLICommand.Parameter.from_string(p) for p in params]

        # assert that positional parameters come before named ones
        met_named = False
        for i, param in enumerate(raw_params):
            if not param.is_named:
                if met_named:
                    raise ValueError(
                        'Positional parameters must come before named ones')
            else:
                met_named = True

        # seperate positional and named parameters
        positional_params = list(map(lambda p: p.value,
                                 filter(lambda x: not x.is_named, raw_params)))
        named_params = dict(map(lambda p: (p.name, p.value),
                                filter(lambda x: x.is_named, raw_params)))

        return CLICommand(command, positional_params, named_params)

    @dataclass
    class Parameter:
        name: str
        value: object

        @property
        def is_named(self) -> bool:
            return self.name is not None

        @staticmethod
        def from_string(param: str) -> 'CLICommand.Parameter':
            if '=' in param:
                name, value = param.split('=')
                return CLICommand.Parameter(name, value)
            else:
                return CLICommand.Parameter(None, param)
