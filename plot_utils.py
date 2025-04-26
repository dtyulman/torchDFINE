import torch
import numpy as np
import matplotlib as mpl
import mpl_toolkits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from python_utils import int_sqrt

@torch.no_grad()
def plot_vs_time(seq, target=None, t_on=None, t_off=None, mode='tiled', max_N=10, max_B=10, ax=None, varname='dim', legend=True, color=None, label=None):
    """
    seq: [b,t,n]
    target: [b,n]
    """
    seq = _prep_mat(seq)
    B,T,N = seq.shape
    B,N = min(B,max_B), min(N,max_N)
    seq = seq[:B,:,:N]

    if target is None:
        target = torch.full((B,N), torch.nan)
    target = _prep_mat(target, num_dims=2)
    target = target[:B,:N]

    if mode == 'overlaid':
        fig, axs = _plot_overlaid(seq, target, ax=ax, legend=legend)
    elif mode == 'tiled':
        fig, axs = _plot_tiled(seq, target, ax=ax, legend=legend, color=color, label=label)
    else:
        raise ValueError('Invalid mode')

    for ax in axs.flatten():
        if t_on is not None and t_on > 0:
            ax.axvline(t_on, color='k', ls='--')
        if t_off is not None and t_off < T:
            ax.axvline(t_off, color='k', ls='--')

    for ax in np.atleast_2d(axs)[-1,:]:
        ax.xaxis.set_tick_params(which="both", labelbottom=True)
        ax.set_xlabel('Time')

    for r,ax in enumerate(np.atleast_2d(axs)[:,0]):
        dimnum = f'_{r}' if mode == 'tiled' and N>1 else ''
        ax.set_ylabel(f'${varname}{dimnum}(t)$')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    return fig, axs


@torch.no_grad()
def _plot_tiled(seq, target=None, ax=None, legend=True, color=None, label=None):
    """Helper for plot_vs_time"""
    B,T,N = seq.shape

    fig, ax = _prep_axes(ax, nrows=N, ncols=B, sharex=True, sharey='row', squeeze=False, figsize=(1.5*B+1, 0.7*N+1))
    for r in range(N):
        for c in range(B):
            lines = ax[r,c].plot(seq[c,:,r], color=color, label=label)
            if target is not None:
                ax[r,c].axhline(target[c,r], color=lines[0].get_color(), ls='--')
    if legend and label:
        ax[0,0].legend()

    # for c in range(B):
    #     ax[-1,c].xaxis.set_tick_params(labelbottom=True)

    return fig, ax


@torch.no_grad()
def _plot_overlaid(seq, target=None, ax=None, legend=True):
    """Helper for plot_vs_time"""
    B,T,N = seq.shape

    fig, ax = _prep_axes(ax, nrows=1, ncols=B, sharex=True, sharey='row', squeeze=False)
    ax = ax.squeeze(0)
    for c in range(B):
        for r in range(N):
            lines = ax[c].plot(seq[c,:,r])
            if target is not None:
                ax[c].axhline(target[c,r], color=lines[0].get_color(), ls='--')

    if legend:
        dummy_lines = [Line2D([], [], color='k', ls='-', label='Value'),
                       Line2D([], [], color='k', ls='--', label='Target')]
        ax[0].legend(dummy_lines, [l._label for l in dummy_lines])

    return fig, ax


@torch.no_grad()
def plot_parametric(seq, t_on=None, t_off=None, mode='line', size=None, cbar=True, ax=None, varname='dim', title=None, cmap='turbo'):
    """
    seq is [T,N] or [B,T,N], where N>=2. Plots only first 3 dimensions if N>3
    """
    seq = _prep_mat(seq)
    B,T,N = seq.shape

    assert N >= 2, 'Use plot_vs_time for 1D signals'
    if N > 3:
        print('Plotting only first 3 dimensions, use plot_vs_time to plot higher dimensional signals over time')
        seq = seq[:,:,:3]

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
    cmap = plt.get_cmap(cmap)
    color = np.vstack([np.tile(default, (t_on, 1)), #[0 ... t_on-1]
                       cmap(np.linspace(0, 1, T_ctrl)), #[t_on ... t_off]
                       np.tile(default, (T-1-t_off, 1))]) #[t_off+1 ... seq_len-1]

    #plot
    line_spec = mode.split('_')
    mode = line_spec[0]
    linestyle = line_spec[1] if len(line_spec) == 2 else 'solid'
    for s in seq:
        if mode == 'line':
            for i in range(T-1):
                ax.plot(*s[i:i+2].T, color=color[i], lw=size, linestyle=linestyle)
        elif mode == 'scatter':
            ax.scatter(*s.T, c=color, s=size)

    #labels
    if cbar:
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(t_on, t_off), cmap=cmap), ax=ax,
                     label='Time', ticks=[t_on, t_off])

    if varname is not None:
        ax.set_xlabel(f'${varname}_1$')
        ax.set_ylabel(f'${varname}_2$')
        if N>=3:
            ax.set_zlabel(f'${varname}_3$')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


@torch.no_grad()
def plot_heatmap(M, h_axis, v_axis, vmin=0, vmax=1, cmap='Reds'):
    #fix cmap
    cmap = mpl.colormaps[cmap]
    cmap.set_over('grey')

    #plot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(h_axis, v_axis, M.T, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis('equal')
    plt.colorbar(pcm, ax=ax, extend='max' if vmax is not None else None)

    return fig, ax


@torch.no_grad()
def plot_eigvals(mat, ax=None, title='', verbose=False, return_eigvals=False):
    fig, ax = _prep_axes(ax)
    mat = _prep_mat(mat)
    eigvals = torch.linalg.eigvals(mat)
    ax.scatter(eigvals.real, eigvals.imag, marker='x')

    x = torch.linspace(0, 2*torch.pi, 100)
    ax.plot(torch.cos(x), torch.sin(x), color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.axis('equal')

    ax.set_xlabel('$Re(\\lambda)$')
    ax.set_ylabel('$Im(\\lambda)$')
    ax.set_title(title)

    if verbose:
        print(f'eig = {eigvals[eigvals.abs()>1e-14].numpy()}')

    if return_eigvals:
        return fig, ax, eigvals
    return fig, ax


@torch.no_grad()
def plot_high_dim(x, d=2, axs=None, label=None, varname='dim', same_color=False, **kwargs):
    """
        x: [b,nx], nx>3
        d: 2 or 3, dimension of each slice through high-dim x
    """
    B,N = x.shape

    if axs is None:
        fig = plt.figure(figsize=(15,8))
        assert d==2 or d==3
    else:
        fig = axs[0].get_figure()
        d = 3 if isinstance(axs[0], mpl_toolkits.mplot3d.axes3d.Axes3D) else 2

    num_subplots = int(np.ceil(N/d))
    rows, cols = int_sqrt(num_subplots)
    for i in range(num_subplots):
        ax = fig.add_subplot(rows,cols,i+1, projection='3d' if d==3 else None) if axs is None else axs[i]
        dims = slice(d*i, min(d*i+d, N))

        color = ax.collections[-1].get_facecolor() if same_color else kwargs.pop('color', None)
        ax.scatter(*x[:,dims].T, label=label, color=color, **kwargs)

        if axs is None:
            ax.set_xlabel(f'${varname}_{{' f'{d*i}' '}$')
            ax.set_ylabel(f'${varname}_{{' f'{d*i+1}' '}$')
            if d==3:
                ax.set_zlabel(f'${varname}_{{' f'{d*i+2}' '}$')

        if i == 0:
            ax.legend()

    fig.tight_layout()
    return fig, fig.get_axes()


###########
# Helpers #
###########
def _prep_axes(ax=None, nrows=1, ncols=1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, **kwargs)

    else:
        # unsqueeze ax into an array
        if not kwargs['squeeze']:
            ax = np.atleast_2d(ax)

        # sanity check if specifying nrows or ncols
        if ncols>1 or nrows>1:
            assert len(ax.flatten()) == nrows*ncols

        # get figure instance
        if isinstance(ax, np.ndarray):
            fig = ax.flatten()[0].get_figure()
        else:
            fig = ax.get_figure()

    return fig, ax


def _prep_mat(mat, num_dims=3, append_dim=0):
    if isinstance(mat, torch.Tensor):
        mat = mat.detach()
    while len(mat.shape) < num_dims:
        if isinstance(mat, torch.Tensor):
            mat = mat.unsqueeze(append_dim)
        elif isinstance(mat, np.ndarray):
            mat = np.expand_dims(mat, append_dim)
        else:
            raise NotImplementedError()
    return mat


def subplots_square(num_subplots, rows=None, cols=None, force=False, **kwargs):
    """Generates an approximately-square grid of n subplots
    """
    if num_subplots > 400 and force==False:
        raise RuntimeWarning("Too many plots ({num_subplots}). To override, pass 'force=True' as second argument.")

    rows, cols = int_sqrt(num_subplots, r=rows, c=cols)
    fig, ax = plt.subplots(rows,cols, **kwargs)
    return fig, ax
