import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_parametric(seq, t_on=None, t_off=None, mode='scatter', add_cbar=True, ax=None, varname=None):
    """
    seq is [T,N], where N>=2. Plots only first 3 dimensions if N>3
    """

    if isinstance(seq, torch.Tensor):
        seq = seq.detach()

    T,N = seq.shape
    assert N >= 2
    if N > 3:
        print('Plotting only first 3 dimensions')
        seq = seq[:,:3]


    #make 2d or 3d axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if N>=3 else None)
    else:
        fig = ax.get_figure()

    #make color sequence for time axis
    if t_on is None: t_on = 0
    if t_off is None: t_off = T-1
    T_ctrl = t_off-t_on+1
    default = [.5, .5, .5, 1.] #RGBA grey
    cmap = plt.cm.jet
    color = np.vstack([np.tile(default, (t_on, 1)), #[0 ... t_on-1]
                       cmap(np.linspace(0, 1, T_ctrl)), #[t_on ... t_off]
                       np.tile(default, (T-1-t_off, 1))]) #[t_off+1 ... seq_len-1]

    #plot
    if mode == 'line':
        for i in range(T-1):
            ax.plot(*seq[i:i+2].T, color=color[i])
    elif mode == 'scatter':
        ax.scatter(*seq.T, c=color)

    #labels
    if add_cbar:
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(t_on, t_off), cmap=cmap), ax=ax,
                     label='Time', ticks=[t_on, t_off])
    if varname is None:
        varname = 'dim'
    ax.set_xlabel(f'${varname}_1$')
    ax.set_ylabel(f'${varname}_2$')
    if N>=3:
        ax.set_zlabel(f'${varname}_3$')

    return fig, ax



###########
# Helpers #
###########
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
