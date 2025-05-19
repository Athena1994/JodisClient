
from core.cli_state_machine.cli_command_page import CLICommandPage
from utils.cli.cli_command import CLICommand


class BaseState:
    def __init__(self, *command_pages: CLICommandPage):
        self._context: dict = None

        self._command_pages = command_pages

    @property
    def context(self) -> dict:
        return self._context

    @context.setter
    def context(self, value: dict):
        self._context = value

    async def handle(self, command: CLICommand) -> 'BaseState':
        for page in self._command_pages:
            if page.has_handler(command.name):
                return await page.handle(self.context, command) or self

        return None

    def prompt_prefix(self) -> str:
        return ''

    def print_help(self):
        print('Available commands:')
        for cmds in [p.commands for p in self._command_pages]:
            for cmd, params in cmds:
                print(f'  {cmd}({params})')
