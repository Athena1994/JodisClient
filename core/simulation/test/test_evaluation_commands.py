

from core.simulation.command_processor import *
import unittest


class DummyCommand:
    def __init__(self, 
                 return_value=None, 
                 loc_var=None, 
                 check: str = None,
                 global_var=None):
        self.executed = False
        self.return_value = return_value
        self.loc_var = loc_var
        self.global_var = global_var
        self.check = check


    def execute(self, env):
        self.executed = True
        if self.return_value is not None:
            env['return'] = self.return_value
        if self.loc_var is not None:
            env['local_vars'].add(self.loc_var)
            env['values'][self.loc_var] = 1
        if self.global_var is not None:
            env['values'][self.global_var] = 1

        if self.check is not None:
            if self.check not in env['values']:
                raise ValueError(f"Variable {self.check} not found")

def mock_create(command):
    return DummyCommand(command.get('return', None),
                        command.get('loc', None),
                        command.get('check', None),
                        command.get('glob', None))     

def patch_mock_create():
    old = CommandFactory.create_command
    def create_command(command):
        if '!' in command and command['!'] == 'dummy':
            return mock_create(command)
        return old(command)
    CommandFactory.create_command = create_command




class TestExpressions(unittest.TestCase):
    
    def test_expressions(self):
        self.assertEqual(Expression("1 + 1").evaluate({}), 2)
        self.assertEqual(Expression("1 - 1").evaluate({}), 0)
        self.assertEqual(Expression("1 * 1").evaluate({}), 1)
        self.assertEqual(Expression("1 / 1").evaluate({}), 1)
        self.assertEqual(Expression("1 % 1").evaluate({}), 0)
        self.assertEqual(Expression("2 ** 4").evaluate({}), 16)
        self.assertEqual(Expression("(1 + 1) *4").evaluate({}), 8)           

        self.assertEqual(Expression("1 + {a}").evaluate({'a': 23}), 24)

        try:
            Expression("{a} + 1").evaluate({})
            self.fail("Should raise exception")
        except ValueError:
            pass

class TestCommandFacilities(unittest.TestCase):

    def test_command_factory(self):
        try:
            CommandFactory.create_command({"!": "unknown", 
                                           "name": "a", 
                                           "value": "1 + 1"})
            self.fail("Should raise exception")
        except ValueError:
            pass    
        


    def test_command_sequence(self):
        patch_mock_create()

        env =  {'values': {}, 'local_vars': set()}
        commands = [DummyCommand(), DummyCommand(), DummyCommand()]
        CommandSequence(commands).execute(env)
        self.assertTrue(commands[0].executed)
        self.assertTrue(commands[1].executed)
        self.assertTrue(commands[2].executed)

        env =  {'values': {}, 'local_vars': set()}
        commands = [DummyCommand(), DummyCommand("ret"), DummyCommand()]
        CommandSequence(commands).execute(env)
        self.assertTrue(commands[0].executed)
        self.assertTrue(commands[1].executed)
        self.assertFalse(commands[2].executed)
        self.assertTrue("return" in env)
        self.assertEqual(env['return'], "ret")

        env =  {'values': {}, 'local_vars': set()}
        conf = [
            {"!": "dummy"},
            {"!": "dummy"},
            {"!": "dummy", "return": True},
        ]

        seq = CommandSequence.from_list(conf)
        seq.execute(env)
        commands = seq._commands
        self.assertEqual(len(commands), 3)
        self.assertTrue(commands[0].executed)
        self.assertTrue(commands[1].executed)
        self.assertTrue(commands[2].executed)
        self.assertTrue("return" in env)
        self.assertTrue(env['return'] is True)

    def test_command_processor(self):
        patch_mock_create()

        try:
            CommandProcessor({})
            self.fail("Should raise exception")
        except ValueError:
            pass

        const = {'a': 1, 'b': 2}
        proc = CommandProcessor({
            'constants': const,
            'process': []
        })
        self.assertEqual(proc._constants, const)
        self.assertIsNot(proc._constants, const)

        proc = CommandProcessor({
            'process': []
        })
        self.assertEqual(proc._constants, {})
        

        dummy_program = {
            "process": [
                {"!": "dummy"},
                {"!": "dummy", 'check': 'a'},
                {"!": "dummy", "return": True},
            ]
        }
        try:
            CommandProcessor(dummy_program).execute({})
            self.fail("Should raise exception")
        except ValueError:
            pass

        dummy_program = {
            "constants": {'c': 1},
            "process": [
                {"!": "dummy", 'loc': 'a'},
                {"!": "dummy", 'glob': 'g'},
                {"!": "dummy", 'check': 'a'},
                {"!": "dummy", 'check': 'b'},
                {"!": "dummy", 'check': 'c'},
                {"!": "dummy", 'check': 'g'},
                {"!": "dummy", "return": True},
            ]
        }
        proc = CommandProcessor(dummy_program)
        proc.execute({'b': 2})
        self.assertTrue('a' not in proc._working_env)
        self.assertTrue('b' not in proc._working_env)
        self.assertTrue('c' in proc._working_env)
        self.assertTrue('g' in proc._working_env)

        proc.reset()
        self.assertTrue('g' not in proc._working_env)
        self.assertTrue('c' in proc._working_env)


class TestCommands(unittest.TestCase):

    def test_store_command(self):
        command = CommandFactory.create_command({"!": "store", "name": "a", "value": "1 + 1"})
        self.assertTrue(isinstance(command, StoreCommand))
        self.assertEqual(command._var_name, 'a')
        self.assertEqual(command._value_expr._expression, '1 + 1')

        env = {'values': {}, 'local_vars': set()}
        store = StoreCommand('a', '1 + 1')
        store.execute(env)
        self.assertTrue('a' in env['values'])
        self.assertEqual(env['values']['a'], 2)

        store = StoreCommand('b', "{a} + 1")
        store.execute(env)
        self.assertTrue('b' in env['values'])
        self.assertEqual(env['values']['b'], 3)

        store = StoreCommand('c', '{b} + {a}')
        store.execute(env)
        self.assertTrue('c' in env['values'])
        self.assertEqual(env['values']['c'], 5)
        
        program = {
            "process": [
                {"!": "store", "name": "a", "value": "1 + 1"},
                {"!": "store", "name": "b", "value": "{a} + 1"},
                {"!": "store", "name": "c", "value": "{b} + {a}"},
            ]
        }
        proc = CommandProcessor(program)
        proc.execute({})
        self.assertTrue('a' in proc._working_env)
        self.assertTrue('b' in proc._working_env)
        self.assertTrue('c' in proc._working_env)
        self.assertEqual(proc._working_env['a'], 2)
        self.assertEqual(proc._working_env['b'], 3)
        self.assertEqual(proc._working_env['c'], 5)

    def test_clear_command(self):
        command = CommandFactory.create_command({"!": "clear", "name": "a"})
        self.assertTrue(isinstance(command, ClearCommand))
        self.assertEqual(command._var_name, 'a')

        env = {'values': {'a': 1, 'b': 4}, 'local_vars': set()}
        clear = ClearCommand('a')
        clear.execute(env)
        self.assertTrue('a' not in env['values'])
        self.assertTrue('a' not in env['local_vars'])

        env = {'values': {'a': 1, 'b': 4}, 'local_vars': {'a'}}
        clear = ClearCommand('a')
        clear.execute(env)
        self.assertTrue('a' not in env['values'])
        self.assertTrue('a' not in env['local_vars'])
        self.assertTrue('b' in env['values'])
        self.assertTrue('b' not in env['local_vars'])

        program = {
            "process": [
                {"!": "store", "name": "a", "value": "1 + 1"},
                {"!": "clear", "name": "a"},
            ]
        }
        proc = CommandProcessor(program)
        proc.execute({})
        self.assertTrue('a' not in proc._working_env)

    def test_set_command(self):
        patch_mock_create()

        command = CommandFactory.create_command({"!": "set", "name": "a", "value": "1"})
        self.assertTrue(isinstance(command, SetCommand))
        self.assertEqual(command._var_name, 'a')
        self.assertEqual(command._value_expr._expression, '1')

        env = {'values': {'b': 3}, 'local_vars': set()}
        set_cmd = SetCommand('a', '1 + {b}')
        set_cmd.execute(env)
        self.assertTrue('a' in env['local_vars'])
        self.assertTrue('a' in env['values'])
        self.assertEqual(env['values']['a'], 4)

        program = {
            "process": [
                {"!": "set", "name": "a", "value": "1 + 1"},
                {"!": "dummy", "check": "a"},
            ]
        }
        proc = CommandProcessor(program)
        proc.execute({})
        self.assertTrue('a' not in proc._working_env)

    def test_if_command(self):
        patch_mock_create()

        command = CommandFactory.create_command(
            {
                "!": "if", 
                "condition": "1 == 1", 
                "then": [{"!": "dummy"}]
            }
        )
        self.assertTrue(isinstance(command, IfCommand))
        self.assertEqual(command._condition._expression, '1 == 1')
        self.assertEqual(len(command._then._commands), 1)

        env = {'values': {'a': 2}, 'local_vars': set()}
        if_cmd = IfCommand("1 == 1", CommandSequence([DummyCommand()]), None)
        if_cmd.execute(env)
        self.assertTrue(if_cmd._then._commands[0].executed)

        if_cmd = IfCommand("1 == 0", CommandSequence([DummyCommand()]), None)
        if_cmd.execute(env)
        self.assertFalse(if_cmd._then._commands[0].executed)

        if_cmd = IfCommand("1 == {a}", 
                           CommandSequence([DummyCommand()]),
                           CommandSequence([DummyCommand()]))
        if_cmd.execute(env)
        self.assertFalse(if_cmd._then._commands[0].executed)
        self.assertTrue(if_cmd._else._commands[0].executed)

        if_cmd = IfCommand("not ({a} == 1)", 
                           CommandSequence([DummyCommand()]),
                           CommandSequence([DummyCommand()]))
        if_cmd.execute(env)
        self.assertTrue(if_cmd._then._commands[0].executed)
        self.assertFalse(if_cmd._else._commands[0].executed)

        program = {
            "process": [
                {"!": "if", "condition": "{b} == 1", 
                 "then": [
                    {"!": "set", "name": "a", "value": "1 + 1"}],
                 "else": [
                    {"!": "set", "name": "a", "value": "1 -1"}],
                 },
                {"!": "return", "value": "{a}"},
            ]
        }
        proc = CommandProcessor(program)
        self.assertEqual(proc.execute({'b': 0}), 0)
        self.assertEqual(proc.execute({'b': 1}), 2)
