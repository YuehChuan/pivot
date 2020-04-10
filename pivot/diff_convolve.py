# -*- coding: utf-8 -*-
import numpy as np

"""diffential 1D arrary and convolution, return gradient"""
def diff_convol(x, show=True, conv_length=3, plotName='Original Signal', plotName2='GradSignal',ax=None):

    ky = x
    kx = np.arange(1, x.shape[0] + 1)

    """convolution"""
    grad = np.diff(ky, 1) / np.diff(kx, 1)
    grad = np.convolve(grad, np.ones(conv_length) / conv_length)  # , mode='valid')
    grad = grad[conv_length - 1:-conv_length + 1]

    if show:
        _plot(x,grad,ax,plotName, plotName2)

    return grad





def _plot(x,grad,ax, plotName:str, plotName2:str):
    """Plot results of the detect_cusum function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            plt.suptitle('differential and convolve')
            ax1 = ax.twinx()
            ax.set_xlabel('pixel')

            ax.set_ylabel(plotName, color='blue', alpha=0.75)
            ax.plot(x, alpha=0.75)
            ax.set_ylim(0, max(x)+100)
            ax1.set_ylabel(plotName2, color='orange')
            ax1.plot(grad, alpha=0.75, color='red')
            ax1.set_ylim(-max(abs(x)), max(abs(x)))

            plt.show()




"""get peaks"""
import numba
@numba.jit(nopython=True)
def flatiron(x, alpha=100., beta=1):
    """
    Flatten signal

    Creator: Michael Kazachok
    Source: https://www.kaggle.com/miklgr500/flatiron
    """
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero * (alpha - beta) / alpha + beta * x[i] / alpha
        new_x[i] = x[i] - zero
    return new_x


@numba.jit(nopython=True)
def drop_missing(intersect, sample):
    """
    Find intersection of sorted numpy arrays

    Since intersect1d sort arrays each time, it's effectively inefficient.
    Here you have to sweep intersection and each sample together to build
    the new intersection, which can be done in linear time, maintaining order.

    Source: https://stackoverflow.com/questions/46572308/intersection-of-sorted-numpy-arrays
    Creator: B. M.
    """
    i = j = k = 0
    new_intersect = np.empty_like(intersect)
    while i < intersect.size and j < sample.size:
        if intersect[i] == sample[j]:  # the 99% case
            new_intersect[k] = intersect[i]
            k += 1
            i += 1
            j += 1
        elif intersect[i] < sample[j]:
            i += 1
        else:
            j += 1
    return new_intersect[:k]


@numba.jit(nopython=True)
def _local_maxima_1d_window_single_pass(x, w):
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays

    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = x.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if x[i - 1] < x[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1

            i_right = i_ahead - 1

            f = False
            i_window_end = i_right + w
            while i_ahead < i_max and i_ahead < i_window_end:
                if x[i_ahead] > x[i]:
                    f = True
                    break
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than x[i]
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_right
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1

            # Skip samples that can't be maximum
            i = i_ahead - 1
        i += 1

    # Keep only valid part of array memory.
    midpoints = midpoints[:m]
    left_edges = left_edges[:m]
    right_edges = right_edges[:m]

    return midpoints, left_edges, right_edges


@numba.jit(nopython=True)
def local_maxima_1d_window(x, w=1):
    """
    Find local maxima in a 1D array.
    This function finds all local maxima in a 1D array and returns the indices
    for their midpoints (rounded down for even plateau sizes).
    It is a modified version of scipy.signal._peak_finding_utils._local_maxima_1d
    to include the use of a window to define how many points on each side to use in
    the test for a point being a local maxima.
    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.
    w : np.int
        How many points on each side to use for the comparison to be True
    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    """

    fm, fl, fr = _local_maxima_1d_window_single_pass(x, w)
    bm, bl, br = _local_maxima_1d_window_single_pass(x[::-1], w)
    bm = np.abs(bm - x.shape[0] + 1)[::-1]
    bl = np.abs(bl - x.shape[0] + 1)[::-1]
    br = np.abs(br - x.shape[0] + 1)[::-1]

    m = drop_missing(fm, bm)

    return m


@numba.jit(nopython=True)
def plateau_detection(grad, threshold, plateau_length=5):
    """Detect the point when the gradient has reach a plateau"""

    count = 0
    loc = 0
    for i in range(grad.shape[0]):
        if grad[i] > threshold:
            count += 1

        if count == plateau_length:
            loc = i - plateau_length
            break

    return loc


def get_peaks(
        x,
        window=3,
        visualise=False,
        visualise_color=None,
):
    """
    Find the peaks in a signal trace.
    Parameters
    ----------
    x : ndarray
        The array to search.
    window : np.int
        How many points on each side to use for the local maxima test
    Returns
    -------
    peaks_x : ndarray
        Indices of midpoints of peaks in `x`.
    peaks_y : ndarray
        Absolute heights of peaks in `x`.
    x_hp : ndarray
        An absolute flattened version of `x`.
    """
    from matplotlib import pyplot as plt



    x_hp = flatiron(x, 100, 1)

    x_dn = np.abs(x_hp)

    peaks = local_maxima_1d_window(x_dn, window)

    heights = x_dn[peaks]

    ii = np.argsort(heights)[::-1]

    peaks = peaks[ii]
    heights = heights[ii]

    ky = heights
    kx = np.arange(1, heights.shape[0] + 1)

    conv_length = 9

    grad = np.diff(ky, 1) / np.diff(kx, 1)
    grad = np.convolve(grad, np.ones(conv_length) / conv_length)  # , mode='valid')
    grad = grad[conv_length - 1:-conv_length + 1]

    knee_x = plateau_detection(grad, -0.01, plateau_length=1000)
    knee_x -= conv_length // 2

    if visualise:
        plt.plot(grad, color=visualise_color)
        #plt.axvline(knee_x, ls="--", color=visualise_color)

    peaks_x = peaks[:knee_x]
    peaks_y = heights[:knee_x]

    ii = np.argsort(peaks_x)
    peaks_x = peaks_x[ii]
    peaks_y = peaks_y[ii]

    return peaks_x, peaks_y, x_hp

