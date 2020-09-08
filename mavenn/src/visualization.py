# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pdb

# Special plotting-related imports
from matplotlib.colors import DivergingNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check


@handle_errors
def _get_45deg_mesh(mat):
    """
    Helper function for heatmap_pairwise. Creates X and Y
    grids, based on mat, rotated -45 degrees.
    """

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
def tidy_df_to_logomaker_df(df,
                            l_col="l",
                            c_col="c",
                            value_col="value"):
    """
    Converts a tidy dataframe of additive params to matrix format.
    df must be a pd.DataFrame that contains columns "l","c","values".
    """
    df = df.copy()
    check(isinstance(df, pd.DataFrame),
          f'type(df)={type(df)}; must be pd.DataFrame.')
    check(l_col in df.columns,
          f'df.columns={df.columns} does not contain "{l_col}".')
    check(c_col in df.columns,
          f'df.columns={df.columns} does not contain "{c_col}".')
    check(value_col in df.columns,
          f'df.columns={df.columns} does not contain "{value_col}".')
    df = df.pivot(index=l_col, columns=c_col, values=value_col)
    df.columns.name = None
    return df


@handle_errors
def heatmap(df,
            l_col="l",
            c_col="c",
            value_col="value",
            missing_values=np.nan,
            mask_dict=None,
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
    Draws a heatmap illustrating a matrix of values.

    parameters
    ----------

    df: (pd.DataFrame)
        A dataframe specifying additive values, eg 1pt effects
        or additive parameters.

    l_col: (str)
        Name of df column containing sequence positions. Values must be
        0,...,L-1.

    c_col: (str)
        Name of df column containing sequence characters.

    value_col: (str)
        Name of df column containing values to illustrate.

    missing_values: (bool)
        Value to use when (l,c) pairs are not observed.

    mask_dict: (dict)
        Specifies which characters to mask at specific positions.
        For example, to mask ACGT at position 3 and AG at position 4, set
        mask_dict={3:'ACGT',4:'AG'}

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

    returns
    -------

    ax: (matplotlib.axes.Axes)
        Axes containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar)
        Colorbar object linked to Axes.
    """

    # Copy dataframe
    df = df.copy()

    # Rename columns
    df = df.rename(columns={l_col: "l",
                            c_col: "c",
                            value_col: "value"})

    # Remove indices with missing characters
    ix = [len(c) > 0 for c in df["c"]]
    df = df[ix]

    # Convert to matrix
    df = tidy_df_to_logomaker_df(df)

    # Fill in missing values
    df = df.fillna(missing_values)

    # Flip
    df = df.loc[:, ::-1]

    # Set extent
    C = df.shape[1]
    L = df.shape[0]
    xlim = [-.5, L - .5]
    ylim = [-.5, C - .5]

    # If wt_seq is set
    if seq:

        # Verify wt_seq is valid
        assert isinstance(seq,
                          str), f'type(wt_seq)={type(seq)} is not str.'

        # Verify wt_seq is composed of valid characters
        wt_seq_set = set(seq)
        char_set = set(df.columns)
        assert wt_seq_set <= char_set, f'wt_seq contains the following invalid characters: {wt_seq_set - char_set}'

    # If there is a mask, set values to nan
    if mask_dict is not None:

        # Make sure mask_dict is a dictionary
        check(isinstance(mask_dict, dict),
              f'type(mask_dict)={type(mask_dict)}; must be dict')

        # Check that mask_dict has valid positions
        mask_ls = mask_dict.keys()
        positions = list(range(L))
        check(set(mask_ls) <= set(positions),
              f'mask_dict={mask_dict} contains positions not compatible with length L={L}.')

        # Check that mask_dict has valid characters
        mask_cs = ''.join(mask_dict.values())
        alphabet = df.columns.values
        check(set(mask_cs) <= set(alphabet),
              f'mask_dict={mask_dict} contains characters not compatible with alphabet={alphabet}.')

        # Set masked values of dataframe to np.nan
        for l, cs in mask_dict.items():
            for c in cs:
                df.loc[l,c] = np.nan


    # Set color lims to central 95% quantile
    if clim is None:
        vals = df.values.ravel()
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

        norm = DivergingNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

    # Otherwise, use uncentered colormap
    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Plot heatmap
    x_edges = np.arange(L + 1) - .5
    y_edges = np.arange(C + 1) - .5
    im = ax.pcolormesh(x_edges,
                       y_edges,
                       df.T,
                       shading='flat',
                       cmap=cmap,
                       clim=clim,
                       norm=norm)

    # Mark wt sequence
    if seq:

        if seq_kwargs is None:
            seq_kwargs = {'marker': '.', 'color': 'k', 's': 2}

        chars = list(df.columns)
        x = range(len(seq))
        y = [chars.index(c) for c in seq]
        ax.scatter(x, y, **seq_kwargs)

    # Style plot
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_yticks(range(C))
    ax.set_yticklabels(df.columns, ha='center')

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
def heatmap_pairwise(df,
                     l1_col="l1",
                     c1_col="c1",
                     l2_col="l2",
                     c2_col="c2",
                     value_col="value",
                     missing_values=np.nan,
                     mask_dict=None,
                     seq=None,
                     seq_kwargs=None,
                     ax=None,
                     gpmap_type="auto",
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
    Draws a triangular heatmap illustrating the pairwise
    parameters of a model. Can be used for pairwise or
    neighbor models.

    Note: The resulting plot has aspect ratio of 1 and
    is scaled so that pixels have half-diagonal lengths given by

        half_pixel_diag = 1/(C*2),

    and blocks of characters have half-diagonal lengths given by

        half_block_diag = 1/2

    This is done so that the horizontal distance between positions
    (as indicated by x ticks) is 1.

    parameters
    ----------

    df: (pd.DataFrame)
        A dataframe listing pairwise model parameters. This can be
        obtained using Model.get_gpmap_parameters(which="pairwise").
        Must contain the following columns:
            "l1": position 1; values are 0,...,L-2.
            "c1": character 1.
            "l2": position 2 > position 1; values are 1,...,L-1.
            "c2": character 2.
            "value": indicates value of $\theta_{l1:c1,l2:c2}$

    l1_col: (str)
        Name of df column containing sequence positions 1. Values must be
        0,...,L-1.

    c1_col: (str)
        Name of df column containing sequence characters at position 1.

    l2_col: (str)
        Name of df column containing sequence positions 2. Values must be
        1,...,L.

    c2_col: (str)
        Name of df column containing sequence characters at position 2.

    value_col: (str)
        Name of df column containing values to illustrate.

    missing_values: (bool)
        Value to use when not provided in df.

    mask_dict: (dict)
        Specifies which characters to mask at specific positions.
        For example, to mask ACGT at position 3 and AG at position 4, set
        mask_dict={3:'ACGT',4:'AG'}

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
        Must be "pairwise", "neighbor", or "auto". If
        "pairwise", a B2-shape will be plotted. If "neighbor",
        a string of diamonds will be plotted. If "auto", this
        will be set automatically depending on the contents of
        df.

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

    returns
    -------

    ax: (matplotlib.axes.Axes)
        Axes containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar)
        Colorbar object linked to Axes.
    """

    df = df.rename(columns={
        l1_col: "l1",
        c1_col: "c1",
        l2_col: "l2",
        c2_col: "c2",
        value_col: "value"
    })
    df = df[["l1", "c1", "l2", "c2", "value"]].copy()

    # Get dims
    L = len(set(list(df['l1']) + list(df['l2'])))
    alphabet = list(set(list(df['c1']) + list(df['c2'])))
    alphabet.sort()
    C = len(alphabet)
    K = (L - 1) * C

    # Create new dataframe with all entries
    pos = np.arange(L).astype(int)
    new_df = pd.DataFrame()
    new_df["l1"] = np.tile(np.reshape(pos, [L, 1, 1, 1]),
                           [1, C, L, C]).ravel()
    new_df["c1"] = np.tile(np.reshape(alphabet, [1, C, 1, 1]),
                           [L, 1, L, C]).ravel()
    new_df["l2"] = np.tile(np.reshape(pos, [1, 1, L, 1]),
                           [L, C, 1, C]).ravel()
    new_df["c2"] = np.tile(np.reshape(alphabet, [1, 1, 1, C]),
                           [L, C, L, 1]).ravel()
    df = new_df.merge(right=df, on=["l1", "c1", "l2", "c2"], how='outer')

    # Remove invalid pairs of positions
    ix = df["l1"] < df["l2"]
    df = df[ix]

    # Fill in missing values
    df = df.fillna(missing_values)

    # Create subscript columns
    df['sub1'] = df['l1'].astype(str) + ':' + df['c1']
    df['sub2'] = df['l2'].astype(str) + ':' + df['c2']

    # Sort rows
    df.sort_values(by=['l1', 'c1', 'l2', 'c2'], inplace=True)

    # Determine if neighbor or pairwise if gpmap_type is not set.
    l1 = df['l1'].values
    l2 = df['l2'].values
    c1 = df['c1'].values
    c2 = df['c2'].values
    assert np.all(l2 > l1), 'Strange; l2 <= l1 sometimes. Shouldnt happen'

    # If wt_seq is set
    if seq:

        # Verify wt_seq is valid
        assert isinstance(seq,
                          str), f'type(wt_seq)={type(seq)} is not str.'

        # Verify wt_seq is composed of valid characters
        wt_seq_set = set(seq)
        char_set = set(alphabet)
        assert wt_seq_set <= char_set, f'wt_seq contains the following invalid characters: {wt_seq_set - char_set}'

    # If there is a mask, set corresponding values to nan
    if mask_dict is not None:

        # Make sure mask_dict is a dictionary
        check(isinstance(mask_dict, dict),
              f'type(mask_dict)={type(mask_dict)}; must be dict')

        # Check that mask_dict has valid positions
        mask_ls = mask_dict.keys()
        positions = list(range(L))
        check(set(mask_ls) <= set(positions),
              f'mask_dict={mask_dict} contains positions not compatible with length L={L}.')

        # Check that mask_dict has valid characters
        mask_cs = ''.join(mask_dict.values())
        check(set(mask_cs) <= set(alphabet),
              f'mask_dict={mask_dict} contains characters not compatible with alphabet={alphabet}.')

        # Set masked values of dataframe to np.nan
        for l, cs in mask_dict.items():
            for c in cs:
                ix1 = (l1 == l) & (c1 == c)
                ix2 = (l2 == l) & (c2 == c)
                ix1or2 = ix1 | ix2
                df.loc[ix1or2, 'value'] = np.nan

    # If user specifies gpmap_type="auto", automatically determine which
    # type of plot to make
    if gpmap_type == "auto":
        if any(l2 - l1 > 1):
            gpmap_type = "pairwise"
        else:
            gpmap_type = "neighbor"

    # If user specifies gpmap_type="neighbor", remove non-neighbor entries
    # from df
    elif gpmap_type == "neighbor":
        df = df[l2 - l1 == 1].copy()

    # Don't do anything if gpmap_type="pairwise"
    elif gpmap_type == "pairwise":
        pass

    # Otherwise, throw an error
    else:
        check(False, f"Invalid gpmap_type={gpmap_type}")

    # Pivot to matrix LCxLC in size
    mat_df = df.pivot(index="sub1", columns="sub2", values="value")
    mat = mat_df.values

    # Verify that mat is the right size
    assert mat.shape == (K, K), f'mat.shape={mat.shape}; expected{(K,K)}'

    # Get indices of finite elements of mat
    ix = np.isfinite(mat)

    # Set color lims to central 95% quantile
    if clim is None:
        vals = mat[ix]
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

        norm = DivergingNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

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
