import time


class Watch:

    @staticmethod
    def get_s(start_ns: int, end_ns: int, digits: int = None) -> float:
        t = (end_ns - start_ns) / 1e9
        if digits is not None:
            t = round(t, digits)
        return t

    @staticmethod
    def scale(t: float, scale: int, digits: int) -> float:
        t *= scale
        if digits is not None:
            t = round(t, digits)
        return t

    def __init__(self):
        self._start = None
        self._cnt = -1
        self._av_time = {'final': 0}

        self._order = []

        self._last_stop = None

    def elapsed(self, digits: int = 2) -> float:
        return self.get_s(self._start, time.time_ns(), digits)

    def get_av_time(self, key: str, digits: int = None, scale: int = 1)\
            -> float:
        return self.scale(self._av_time[key][0], scale, digits)

    def reset(self):
        self._start = None
        self._cnt = -1
        self._av_time = {'final': (0, 0)}
        self._order = []
        self._last_stop = None

    def start(self):
        self._start = time.time_ns()
        self._cnt += 1
        self._last_stop = self._start

    def stop(self,
             key: str,
             since_beginning: bool = False,
             final: bool = False,
             return_average: bool = False) -> float:

        s = self._start if since_beginning else self._last_stop
        self._last_stop = time.time_ns()

        if final:
            t = self.get_s(self._start, self._last_stop)
            self._av_time['final'] = (
                (self._av_time['final'][0] * self._cnt + t) / (self._cnt + 1),
                self._cnt + 1)

        t = self.get_s(s, self._last_stop)
        if key not in self._av_time:
            self._av_time[key] = (t, 1)
            self._order.append(key)
        else:
            val, num = self._av_time[key]
            self._av_time[key] = (((val * num) + t) / (num + 1), num + 1)

        if return_average:
            return t, self._av_time[key][0]
        else:
            return t

    def print(self, title: str, scale: int = 1, digits: int = 4):

        total = self.scale(self._av_time['final'][0], scale, digits)
        print(f"{title} (total: {total} s):")

        for key in self._order:
            t = self.scale(self._av_time[key][0], scale, digits)
            print(f"-- {key}: {t} s ({t / total * 100:.2f}%)")

    def get_it(self) -> int:
        return self._cnt + 1
