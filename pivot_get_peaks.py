# -*- coding: utf-8 -*-
from pivot.load_data import load_txt
from pivot.diff_convolve import get_peaks
from matplotlib import pyplot as plt


input_np = load_txt("value.txt")

px, py, _ = get_peaks(input_np, window=3, visualise=True, visualise_color=None)
print(px)  # debug

"""show peaks"""
plt.xlabel('pixel')
plt.ylabel('gray level')
plt.suptitle('denoised peak indexing')

plt.ylim((0, 200))  # add feature :autoscale with max value
plt.plot(input_np, color='blue', alpha=0.75)
plt.scatter(px, input_np[px], color="red")
plt.show()

