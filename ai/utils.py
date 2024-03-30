class Actions:
    WAIT = 0
    BUY = 1
    SELL = 2

    @staticmethod
    def get_all():
        return [v for v in vars(Actions) if v[0:2] != '__' and not callable(Actions.__dict__[v])]

    @staticmethod
    def get(ix: int):
        items = Actions.get_all()
        return Actions.__dict__[items[ix]]

