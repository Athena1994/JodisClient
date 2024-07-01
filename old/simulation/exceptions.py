
class BuyFailedException(Exception):
    def __init__(self, txt=None):
        super().__init__(txt)


class SellFailedException(Exception):
    def __init__(self, txt=None):
        super().__init__(txt)
