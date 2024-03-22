import numpy as np
import matplotlib.pyplot as plt


def prep_axes(ax=None, nrows=1, ncols=1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, **kwargs)
    else:
        if ncols>1 or nrows>1: #sanity check if specifying nrows or ncols
            assert len(ax.flatten()) == nrows*ncols

        if isinstance(ax, np.ndarray):
            fig = ax[0].get_figure()
        else:
            fig = ax.get_figure()
    return fig, ax
