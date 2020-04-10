# -*- coding: utf-8 -*-
import unittest
from pivot.load_data import load_txt,load_1DViewer_list
from pivot.diff_convolve import diff_convol, get_peaks
from pivot.detect_cusum import detect_cusum


class MyPivot(unittest.TestCase):
    def test_load_txt(self):
        f = open("value.txt", "r")
        input=[]
        for x in f.readlines():
            x = x.strip('\n')
            input.append(int(x))

        f.close()
        return input

    def test_pivot(self, show=True):
        input_np=load_txt("value.txt")
        diffential_input=diff_convol(input_np)
        taMv, taiMv, tafMv, ampMv = detect_cusum(diffential_input, 4, 3, True, show)

    def test_get_peaks(self):
        from matplotlib import pyplot as plt
        input_np=load_txt("value.txt")

        px, py, _ = get_peaks(input_np, window=3, visualise=True,visualise_color=None )
        print(px)  # debug

        """show peaks"""
        plt.xlabel('pixel')
        plt.ylabel('gray level')
        plt.suptitle('denoised peak indexing')

        plt.ylim((0, 200))  # add feature :autoscale with max value
        plt.plot(input_np, color='blue',alpha=0.75)
        plt.scatter(px, input_np[px], color="red")
        plt.show()

    def load_1DViewer_list(self):
        from pivot.viewer_list import Viewer
        list=Viewer('medBlur.jpg')

        return list

    def pivot_load_1DViewer(self, show=True):
        from pivot.load_data import load_1DViewer_list
        input_np=load_1DViewer_list('medBlur.jpg')
        diffential_input=diff_convol(input_np)
        taMv, taiMv, tafMv, ampMv = detect_cusum(diffential_input, 4, 3, True, show)







if __name__ == '__main__':
    unittest.main()
