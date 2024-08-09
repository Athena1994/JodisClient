
import shutil
import tempfile


class TmpDir:
    def __init__(self):
        self.path = None

    def __enter__(self):
        return self.create()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def get(self) -> str:
        return self.path

    def close(self):
        shutil.rmtree(self.path)
        self.path = None
