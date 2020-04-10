# -*- coding: utf-8 -*-
from pivot.load_data import load_txt
from pivot.diff_convolve import diff_convol
from pivot.detect_cusum import detect_cusum


def pivot(show=True):
    input_np = load_txt("value.txt")
    diffential_input = diff_convol(input_np)
    taMv, taiMv, tafMv, ampMv = detect_cusum(diffential_input, 4, 3, True, show)


if __name__ == '__main__':
        pivot(show=True)
