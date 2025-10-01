import healpy as hp
# import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.transforms import blended_transform_factory
# import pysm3.units as u

from cmbml.core.asset_handlers import HealpyMap
from cmbml.utils.planck_cmap import colombi1_cmap
from cmbml.utils.cmap_interp import get_symlog_cmap  #, get_log_cmap, get_linear_cmap
from cmbml.utils.physics_mask import downgrade_mask
from .number_plot_ticks import make_tick_labels, generate_symlog_ticks


def maybe_disable_symlog(min_val, max_val, linthresh, k=3.0):
    """
    Returns a (possibly modified) linthresh.
    If the data range is within ±(k x linthresh), we don't need symlog,
    so we return a large linthresh to flatten the log parts.

    Otherwise, return the original linthresh.
    """
    if min_val >= 0:
        max_dist = max_val
    elif max_val <= 0:
        max_dist = -min_val
    else:
        max_dist = max(-min_val, max_val)

    if max_dist <= k * linthresh:
        return max_dist * 1.01  # bump it slightly above max_val to fully suppress log
    else:
        return linthresh


def show_obs_logNxN(maps, labels, unit=None, mask=None, min_val=-1e3, max_val=1e7, linthresh=400, linscale=1, n_cols=3, n_rows=3, dpi=100, xsize=200, suptitle="", unit_label=None, out_fn=None):
    if linscale is None:
        linscale = 1
    if linthresh is None:
        linthresh = 400
    linthresh = maybe_disable_symlog(min_val, max_val, linthresh, k=3.0)
    
    figsize = (n_cols*5, n_rows*2.5 + 1)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, figure=fig, hspace=-0.1, wspace=0.05)

    norm = cm.colors.SymLogNorm(vmin=min_val, vmax=max_val, linthresh=linthresh * 0.98, linscale=linscale)
    symlog_cmap = get_symlog_cmap(colombi1_cmap, norm)
    plot_params = dict(min=0, max=1, xsize=xsize, unit="$\\mu K_{CMB}$", cbar=False, cmap=symlog_cmap, hold=True)

    # Store the positions of each subplot in figure coordinates
    axes_positions = []

    for i in range(len(maps)):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        plt.axes(ax)
        to_show = norm(maps[i])
        if mask is not None:
            to_show = hp.ma(to_show)
            to_show.mask = mask
        hp.mollview(to_show, title=labels[i], **plot_params)

        # Get the subplot position in figure coordinates
        pos = ax.get_position()
        axes_positions.append(pos)

    # Add colorbar
    c_height_dict = {
        (3,3): 0.10,
        (2,2): 0.12,
        (2,1): 0.20,
        (3,1): 0.20,
        (3,2): 0.12,
        (1,1): 0.18,
    }
    cax = fig.add_axes([0.2, 
                        c_height_dict[(n_cols, n_rows)], 
                        0.6, 
                        0.015])
    mappable = cm.ScalarMappable(norm=norm, cmap=symlog_cmap)
    cb = plt.colorbar(mappable, cax=cax, orientation='horizontal')
    if unit_label is None:
        try:
            unit = maps[0].unit
            unit = unit.to_string('latex_inline')
            unit_label = "$\\delta \\text{T} \\; $[" + unit + ']'
        except:
            unit_label = ""
    cb.set_label(unit_label)

    ticks = generate_symlog_ticks(min_val=min_val,
                                  max_val=max_val,
                                  linthresh=linthresh,
                                  linscale=linscale,
                                  )
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticks([], minor=True)
        cb.set_ticklabels(make_tick_labels(ticks))

    trans = blended_transform_factory(cax.transData, cax.transAxes)

    if min_val < -linthresh:
        cax.plot([-linthresh], [0.5], marker='4', color='black', markersize=8, transform=trans, clip_on=False)
    if max_val > linthresh:
        cax.plot([linthresh], [0.5], marker='3', color='black', markersize=8, transform=trans, clip_on=False)

    sup_height = {
        1: 0.63,
        2: 0.73,
        3: 0.77
    }

    if suptitle:
        fig.text(
            0.5,                      # Centered horizontally
            cax.get_position().y1 + sup_height[n_rows],  # Just above the colorbar
            suptitle,
            ha='center',
            va='bottom',
            fontsize=plt.rcParams['axes.titlesize']
        )

    if out_fn is None:
        plt.show()
    else:
        plt.savefig(out_fn)
    plt.close()


def get_mask(nside, mask_fn):
    h = HealpyMap()
    m = h.read(mask_fn, map_fields=[3])[0]
    m = downgrade_mask(m, nside, 0.9)
    return 1-m


