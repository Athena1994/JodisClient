

from pathlib import Path
import unittest

from utils.requirements_check import PipRequirementsControl


class PipControlTest(unittest.TestCase):
    def setUp(self):
        self.pip_control = PipRequirementsControl()

    def test_install_requirements(self):
        # Test with a valid requirements file
        requirements_file = "assets/requirements.txt"

        print(Path(requirements_file).exists())
