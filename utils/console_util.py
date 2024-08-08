import sys

import numpy as np


class ScreenBuilder:

    def __init__(self):
        self.current_line = 0
        self.lowest_line = 0
        self.marks = {}

        self.print_box = False
        self.box_width = None

    def print_line(self, *txt, go_to_end=True, clear_line=True):

        if clear_line:
            self.clear_line()

        self.print(*txt)
        sys.stdout.write('\n')
        self._set_current_line(self.current_line+1)
        if go_to_end:
            self.goto_end()

    def _set_current_line(self, val):
        self.current_line = val
        if self.lowest_line < self.current_line:
            self.lowest_line = self.current_line

    def clear_line(self):
        sys.stdout.write('\x1B[K')

    def print(self, *txt, cr=False):
        sys.stdout.write(('\r' if cr else '')
                         + (' '.join([t if type(t) is str else repr(t)
                                      for t in txt])))

    def reset_position(self):
        sys.stdout.write('\x1B[0;0H')
        self.current_line = 0

    def go_up(self, line_cnt):
        sys.stdout.write(f'\r\x1B[{line_cnt}A')
        self._set_current_line(self.current_line-line_cnt)

    def go_down(self, line_cnt):
        sys.stdout.write(f'\r\x1B[{line_cnt}B')
        self._set_current_line(self.current_line+line_cnt)

    def goto_line(self, y):
        if y < self.current_line:
            self.go_up(self.current_line - y)
        elif y > self.current_line:
            self.go_down(y - self.current_line)

    def mark_line(self, name):
        self.marks[name] = self.current_line

    def goto_mark(self, name):
        self.goto_line(self.marks[name])

    def remove_mark(self, name):
        if name in self.marks:
            self.marks.pop(name)

    def has_mark(self, name):
        return name in self.marks

    def hide_cursor(self):
        sys.stdout.write('\x1B[? 25l')

    def show_cursor(self):
        sys.stdout.write('\x1B[? 25h')

    def goto_end(self):
        self.goto_line(self.lowest_line)


class TableBuilderEx:

    # if name is not None, table will always print to the same location
    def __init__(self, sb: ScreenBuilder, name: str = None):
        self.max_line_len = 0

        self.name = name

        self.inter = ' │ '
        self.vl = '│'
        self.hl = '─'

        self.blocks = [None]
        self.block_sizes = []

        self.sb = sb

    def new_block(self):
        self.blocks.append(None)

    def get_line_len(self, block):
        return len(self.inter) * (len(self.block_sizes[block])-1) \
            + np.sum(self.block_sizes[block])

    def add_line(self, *args):
        if self.blocks[-1] is None:
            self.blocks[-1] = [list(args)]
            self.block_sizes.append(np.array([len(s) for s in args]))
        else:
            if len(self.block_sizes[-1]) != len(args):
                raise Exception(f'collumn number ({len(args)}) does not match '
                                f'current block ({len(self.blocks[-1])})')
            str_args = [s if type(s) is str else repr(s) for s in args]

            self.blocks[-1].append(list(str_args))
            cell_lengths = np.array([len(s) for s in str_args])
            self.block_sizes[-1] = np.maximum(cell_lengths,
                                              self.block_sizes[-1])

    def print_hline(self, p):
        if p == -1:
            self.sb.print_line('┌' + self.hl * (self.max_line_len) + '┐',
                               go_to_end=False)
        if p == 0:
            self.sb.print_line('├' + self.hl * (self.max_line_len) + '┤',
                               go_to_end=False)
        if p == 1:
            self.sb.print_line('└' + self.hl * (self.max_line_len) + '┘')

    @staticmethod
    def adjust_element(txt, target_len):
        return txt + ' '*(target_len-len(txt))

    def build_line(self, elements, block_sizes):
        line = self.inter.join([self.adjust_element(el, bs)
                                for el, bs in zip(elements, block_sizes)])
        return self.vl \
            + line + ' '*(self.max_line_len-len(line))\
            + self.vl

    def print(self, reset_loc=False):

        if self.name is not None:
            if (not reset_loc) and self.sb.has_mark(self.name):
                self.sb.goto_mark(self.name)
            else:
                self.sb.mark_line(self.name)

        self.max_line_len = max([self.get_line_len(i)
                                 for i in range(len(self.blocks))])

        self.print_hline(-1)
        for i, (lines, target_size) in enumerate(zip(self.blocks,
                                                     self.block_sizes)):
            if i != 0:
                self.print_hline(0)

            for line in lines:
                self.sb.print_line(self.build_line(line, target_size),
                                   go_to_end=False)

        self.print_hline(1)


def print_progress_bar(prefix, i, cnt, suffix, fill_char='█', bar_length=50,
                       sb: ScreenBuilder = None, name='bar', reset_loc=False):

    perc = i/cnt
    filled = round(perc*bar_length)
    not_filled = bar_length-filled
    bar = fill_char*filled + '-'*not_filled
    txt = f'\r{prefix}:\t|{bar}| ({i}/{cnt} | {round(perc*100)}%) - {suffix}'

    if sb is None:
        print(txt, end='', flush=True)
    else:
        if reset_loc or (not sb.has_mark(name)):
            sb.mark_line(name)
        else:
            sb.goto_mark(name)
        sb.print_line(txt)
