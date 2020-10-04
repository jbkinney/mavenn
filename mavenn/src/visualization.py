"""visualization.py: Functions for visualizing MAVE-NN models."""
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pdb

# Special plotting-related imports
#from matplotlib.colors import DivergingNorm, Normalize
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_alphabet, validate_seqs

@handle_errors
def _get_45deg_mesh(mat):
    """Create X and Y grids rotated -45 degreees."""
    # Define rotation matrix
    theta = -np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Define unrotated coordinates on
    K = len(mat) + 1
    grid1d = np.arange(0, K) - .5
    X = np.tile(np.reshape(grid1d, [K, 1]), [1, K])
    Y = np.tile(np.reshape(grid1d, [1, K]), [K, 1])
    xy = np.array([X.ravel(), Y.ravel()])

    # Rotate coordinates
    xy_rot = R @ xy
    X_rot = xy_rot[0, :].reshape(K, K)
    Y_rot = xy_rot[1, :].reshape(K, K).T

    return X_rot, Y_rot


@handle_errors
def heatmap(values,
            alphabet,
            seq=None,
            seq_kwargs=None,
            ax=None,
            show_spines=False,
            cbar=True,
            cax=None,
            clim=None,
            clim_quantile=1,
            ccenter=None,
            cmap='coolwarm',
            cmap_size="5%",
            cmap_pad=0.1):
    """
    Draw a heatmap illustrating a matrix of values.

    Parameters
    ----------
    values: (np.ndarray)
        Array sized (L,C) that contains values to plot

    alphabet: (np.ndarray)
        1D array containing characters in alphabet.

    seq: (str)
        The sequence to show, if any. Must have length len(df)
        and be comprised of characters in df.columns.

    seq_kwargs: (dict)
        Arguments to pass to plt.plot() when illustrating seq characters.

    ax: (matplotlib.axes.Axes)
        The Axes object on which the heatmap will be drawn.
        If None, one will be created. If specified, cbar=True,
        and cax=None, ax will be split in two to make room for
        colorbar.

    show_spines: (bool)
        Whether to show axes spines.

    cbar: (bool)
        Whether to draw a colorbar.

    cax: (matplotlib.axes.Axes)
        The Axes object on which the colorbar will be drawn
        if requested. If None, one will be created by splitting
        ax in two according to cmap_size and cmpa_pad.

    clim: (array of form [cmin, cmax])
        Optional specification of the maximum and minimum effect
        values spanned by the colormap. Overrides clim_quantile.

    clim_quantile: (float in [0,1])
        If set, clim will automatically chosen to include the specified
        fraction of effect sizes.

    ccenter: (float)
        The effect value at which to position the center of a diverging
        colormap. A value of ccenter=0 often makes sense.

    cmap: (str or matplotlib.colors.Colormap)
        Colormap to use.

    cmap_size: (str)
        Specifies the fraction of ax width used for colorbar.
        See documentation for
            mpl_toolkits.axes_grid1.make_axes_locatable().

    cmap_pad: (float)
        Specifies space between colorbar and shrunken ax.
        See documentation for
            mpl_toolkits.axes_grid1.make_axes_locatable().

    Returns
    -------
    ax: (matplotlib.axes.Axes)
        Axes containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar)
        Colorbar object linked to Axes.
    """
    alphabet = validate_alphabet(alphabet)
    L, C = values.shape

    # Set extent
    xlim = [-.5, L - .5]
    ylim = [-.5, C - .5]

    # If wt_seq is set, validate it.
    if seq:
        seq = validate_seqs(seq, alphabet)

    # Set color lims to central 95% quantile
    if clim is None:
        vals = values.ravel()
        vals = vals[np.isfinite(vals)]
        clim = np.quantile(vals, q=[(1 - clim_quantile) / 2,
                                    1 - (1 - clim_quantile) / 2])

    # Create axis if none already exists
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Needed to center colormap at zero
    if ccenter is not None:

        # Reset ccenter if is not compatible with clim
        if (clim[0] > ccenter) or (clim[1] < ccenter):
            ccenter = 0.5 * (clim[0] + clim[1])

        norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

    # Otherwise, use uncentered colormap
    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Plot heatmap
    x_edges = np.arange(L + 1) - .5
    y_edges = np.arange(C + 1) - .5
    im = ax.pcolormesh(x_edges,
                       y_edges,
                       values.T,
                       shading='flat',
                       cmap=cmap,
                       clim=clim,
                       norm=norm)

    # Mark wt sequence
    _ = np.newaxis
    if seq:

        # Set marker style
        if seq_kwargs is None:
            seq_kwargs = {'marker': '.', 'color': 'k', 's': 2}

        # Get xy coords to plot
        seq_arr = np.array(list(seq[0]))
        xy = np.argwhere(seq_arr[:, _] == alphabet[_, :])

        # Mark sequence
        ax.scatter(xy[:, 0], xy[:, 1], **seq_kwargs)
        #pdb.set_trace()

    # Style plot
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_yticks(range(C))
    ax.set_yticklabels(alphabet, ha='center')
    ax.invert_yaxis()

    if not show_spines:
        for loc, spine in ax.spines.items():
            spine.set_visible(False)

    # Create colorbar if requested, make one
    if cbar:
        if cax is None:
            cax = make_axes_locatable(ax).new_horizontal(size=cmap_size,
                                                         pad=cmap_pad)
            fig.add_axes(cax)
        cb = plt.colorbar(im, cax=cax)

        # Otherwise, return None for cb
    else:
        cb = None

    return ax, cb


@handle_errors
def heatmap_pairwise(values,
                     alphabet,
                     seq=None,
                     seq_kwargs=None,
                     ax=None,
                     gpmap_type="pairwise",
                     show_position=False,
                     position_size=None,
                     position_pad=1,
                     show_alphabet=True,
                     alphabet_size=None,
                     alphabet_pad=1,
                     show_seplines=True,
                     sepline_kwargs=None,
                     xlim_pad=.1,
                     ylim_pad=.1,
                     cbar=True,
                     cax=None,
                     clim=None,
                     clim_quantile=1,
                     ccenter=0,
                     cmap='coolwarm',
                     cmap_size="5%",
                     cmap_pad=0.1):
    """
    Draw a triangular heatmap illustrating the pairwise or neighbor parameters.

    Note: The resulting plot has aspect ratio of 1 and
    is scaled so that pixels have half-diagonal lengths given by

        half_pixel_diag = 1/(C*2),

    and blocks of characters have half-diagonal lengths given by

        half_block_diag = 1/2

    This is done so that the horizontal distance between positions
    (as indicated by x ticks) is 1.

    Parameters
    ----------
    values: (np.array)
        An array, shape (L,C,L,C), containing pairwise parameters.
        Note that only values at coordinates [l1,c1,l2,c2] with l2 > l1
        will be plotted. NaN values will not be plotted.

    alphabet: (np.array)
        Array of shape (C,) containing alphabet characters.

    seq: (str)
        The sequence to show, if any. Must have length len(df)
        and be comprised of characters in df.columns.

    seq_kwargs: (dict)
        Arguments to pass to plt.plot() when illustrating seq characters.

    ax: (matplotlib.axes.Axes)
        The Axes object on which the heatmap will be drawn.
        If None, one will be created. If specified, cbar=True,
        and cax=None, ax will be split in two to make room for
        colorbar.

    gpmap_type: (str)
        Determines how many pairwise parameters are plotted.
        Must be "pairwise", "neighbor". If "pairwise", a B2-bomber shape
        will be plotted. If "neighbor", a string of diamonds will be plotted.

    show_position: (bool)
        Whether to draw position labels on the plot.

    position_size: (float >= 0)
        Font size to use for position labels.

    position_pad: (float)
        Additional padding, in units of half_pixel_diag, used to space
        the position labels further from the heatmap.

    show_alphabet: (bool)
        Whether to draw alphabet on the plot.

    alphabet_size: (float >= 0)
        Font size to use for alphabet.

    alphabet_pad: (float)
        Additional padding, in units of half_pixel_diag, used to space
        the c1 alphabet labels from the heatmap.

    show_seplines: (bool)
        Whether to draw seplines, i.e. lines separating character blocks
        for different position pairs.

    sepline_kwargs: (dict)
        Keywords to pass to ax.plot() when drawing seplines.

    xlim_pad: (float)
        Additional padding to add to both xlims, in absolute units.

    ylim_pad: (float)
        Additional padding to add to both ylims, in aboslute units.

    cbar: (bool)
        Whether to draw a colorbar.

    cax: (matplotlib.axes.Axes)
        The Axes object on which the colorbar will be drawn
        if requested. If None, one will be created by splitting
        ax in two according to cmap_size and cmpa_pad.

    clim: (array of form [cmin, cmax])
        Optional specification of the maximum and minimum effect
        values spanned by the colormap. Overrides clim_quantile.

    clim_quantile: (float in [0,1])
        If set, clim will automatically be chosen to include the specified
        fraction of pixel values.

    ccenter: (float)
        The pixel value at which to position the center of a diverging
        colormap. A value of ccenter=0 most often makes sense.

    cmap: (str or matplotlib.colors.Colormap)
        Colormap to use.

    cmap_size: (str)
        Specifies the fraction of ax width used for colorbar.
        See documentation for
            mpl_toolkits.axes_grid1.make_axes_locatable().

    cmap_pad: (float)
        Specifies space between colorbar and shrunken ax.
        See documentation for
            mpl_toolkits.axes_grid1.make_axes_locatable().

    Returns
    -------
    ax: (matplotlib.axes.Axes)
        Axes containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar)
        Colorbar object linked to Axes.
    """
    # Validate values
    check(isinstance(values, np.ndarray),
          f'type(values)={type(values)}; must be np.ndarray.')
    check(values.ndim==4,
          f'values.ndim={values.ndim}; must be 4.')
    L, C, L2, C2 = values.shape
    check(L2 == L and C2 == C,
          f'values.shape={values.shape} is invalid; must be of form (L,C,L,C.')
    values = values.copy()

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)
    check(len(alphabet) == C,
          f'len(alphabet)={len(alphabet)} does not match C={C}')

    ls = np.arange(L).astype(int)
    l1_grid = np.tile(np.reshape(ls, (L, 1, 1, 1)),
                      (1, C, L, C))
    l2_grid = np.tile(np.reshape(ls, (1, 1, L, 1)),
                      (L, C, 1, C))

    # If user specifies gpmap_type="neighbor", remove non-neighbor entries
    if gpmap_type == "neighbor":
        nan_ix = ~(l2_grid - l1_grid == 1)

    # Don't do anything if gpmap_type="pairwise"
    elif gpmap_type == "pairwise":
        nan_ix = ~(l2_grid - l1_grid >= 1)

    else:
        check(False, f'Unrecognized gpmap_type={repr(gpmap_type)}.')

    # Set values at invalid positions to nan
    values[nan_ix] = np.nan

    # Reshape values into a matrix
    mat = values.reshape((L*C, L*C))
    mat = mat[:-C, :]
    mat = mat[:, C:]
    K = (L - 1) * C

    # Verify that mat is the right size
    assert mat.shape == (K, K), \
        f'mat.shape={mat.shape}; expected{(K,K)}. Should never happen.'

    # Get indices of finite elements of mat
    ix = np.isfinite(mat)

    # Set color lims to central 95% quantile
    if clim is None:
        clim = np.quantile(mat[ix], q=[(1 - clim_quantile) / 2,
                                    1 - (1 - clim_quantile) / 2])

    # Create axis if none already exists
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Needed to center colormap at zero
    if ccenter is not None:

        # Reset ccenter if is not compatible with clim
        if (clim[0] > ccenter) or (clim[1] < ccenter):
            ccenter = 0.5 * (clim[0] + clim[1])

        norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Get rotated mesh
    X_rot, Y_rot = _get_45deg_mesh(mat)

    # Normalize
    half_pixel_diag = 1 / (2*C)
    pixel_side = 1 / (C * np.sqrt(2))
    X_rot = X_rot * pixel_side + half_pixel_diag
    Y_rot = Y_rot * pixel_side


    # Set parameters that depend on gpmap_type
    ysep_min = -0.5 - .001 * half_pixel_diag
    xlim = [-xlim_pad, L - 1 + xlim_pad]
    if gpmap_type == "pairwise":
        ysep_max = L / 2 + .001 * half_pixel_diag
        ylim = [-0.5 - ylim_pad, (L - 1) / 2 + ylim_pad]
    else:
        ysep_max = 0.5 + .001 * half_pixel_diag
        ylim = [-0.5 - ylim_pad, 0.5 + ylim_pad]

    # Not sure why I have to do this
    Y_rot = -Y_rot

    # Draw rotated heatmap
    im = ax.pcolormesh(X_rot,
                       Y_rot,
                       mat,
                       cmap=cmap,
                       norm=norm)

    # Remove spines
    for loc, spine in ax.spines.items():
        spine.set_visible(False)

    # Set sepline kwargs
    if show_seplines:
        if sepline_kwargs is None:
            sepline_kwargs = {'color': 'gray',
                              'linestyle': '-',
                              'linewidth': .5}

        # Draw white lines to separate position pairs
        for n in range(0, K+1, C):

            # TODO: Change extent so these are the right length
            x = X_rot[n, :]
            y = Y_rot[n, :]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

            x = X_rot[:, n]
            y = Y_rot[:, n]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

    # Set lims
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set aspect
    ax.set_aspect("equal")

    # Remove yticks
    ax.set_yticks([])

    # Set xticks
    xticks = np.arange(L).astype(int)
    ax.set_xticks(xticks)

    # If drawing characters
    if show_alphabet:

        # Draw c1 alphabet
        for i, c in enumerate(alphabet):
            x1 = 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            y1 = - 0.5 * half_pixel_diag \
                 - i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            ax.text(x1, y1, c, va='center',
                    ha='center', rotation=-45, fontsize=alphabet_size)

        # Draw c2 alphabet
        for i, c in enumerate(alphabet):
            x2 = 0.5 + 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 + alphabet_pad * half_pixel_diag
            y2 = - 0.5 + 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            ax.text(x2, y2, c, va='center',
                    ha='center', rotation=45, fontsize=alphabet_size)

    # Display positions if requested (only if model is pairwise)
    l1_positions = np.arange(0, L-1)
    l2_positions = np.arange(1, L)
    half_block_diag = C * half_pixel_diag
    if show_position and gpmap_type == "pairwise":

        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = 0.5 * half_block_diag \
                 + i * half_block_diag \
                 - position_pad * half_pixel_diag
            y2 = 0.5 * half_block_diag \
                 + i * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x2, y2, f'{l2:d}', va='center',
                    ha='center', rotation=45, fontsize=position_size)

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = (L - 0.5) * half_block_diag \
                 + i * half_block_diag \
                 + position_pad * half_pixel_diag
            y1 = (L - 1.5) * half_block_diag \
                 - i * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x1, y1, f'{l1:d}', va='center',
                    ha='center', rotation=-45, fontsize=position_size)

    elif show_position and gpmap_type == "neighbor":

        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = 0.5 * half_block_diag \
                 + 2 * i * half_block_diag \
                 - position_pad * half_pixel_diag
            y2 = 0.5 * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x2, y2, f'{l2:d}', va='center',
                    ha='center', rotation=45, fontsize=position_size)

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = 1.5 * half_block_diag \
                 + 2* i * half_block_diag \
                 + position_pad * half_pixel_diag
            y1 = + 0.5 * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x1, y1, f'{l1:d}', va='center',
                    ha='center', rotation=-45, fontsize=position_size)

    # Mark wt sequence
    if seq:

        # Set seq_kwargs if not set in constructor
        if seq_kwargs is None:
            seq_kwargs = {'marker': '.', 'color': 'k', 's': 2}

        # Iterate over pairs of positions
        for l1 in range(L):
            for l2 in range(l1+1, L):

                # Break out of loop if gmap_type is "neighbor" and l2 > l1+1
                if (l2-l1 > 1) and gpmap_type == "neighbor":
                    continue

                # Iterate over pairs of characters
                for i1, c1 in enumerate(alphabet):
                    for i2, c2 in enumerate(alphabet):

                        # If there is a match to the wt sequence,
                        if seq[l1] == c1 and seq[l2] == c2:

                            # Compute coordinates of point
                            x = half_pixel_diag + \
                                (i1 + i2) * half_pixel_diag + \
                                (l1 + l2 - 1) * half_block_diag
                            y = (i2 - i1) * half_pixel_diag + \
                                (l2 - l1 - 1) * half_block_diag

                            # Plot point
                            ax.scatter(x, y, **seq_kwargs)


    # Create colorbar if requested, make one
    if cbar:
        if cax is None:
            cax = make_axes_locatable(ax).new_horizontal(size=cmap_size,
                                                         pad=cmap_pad)
            fig.add_axes(cax)
        cb = plt.colorbar(im, cax=cax)

        # Otherwise, return None for cb
    else:
        cb = None

    return ax, cb
