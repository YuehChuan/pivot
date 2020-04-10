# -*- coding: utf-8 -*-
import numpy as np

def load_txt(fileName:str):
    f = open(fileName, "r")
    input = []
    for x in f.readlines():
        x = x.strip('\n')
        input.append(int(x))

    input_np = np.asarray(input, dtype=np.float64)
    f.close()
    return input_np


def load_1DViewer_list(PATH:str):
    from pivot.viewer_list import Viewer
    list = Viewer('medBlur.jpg')

    return list