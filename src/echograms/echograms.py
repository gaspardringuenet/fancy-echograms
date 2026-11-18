import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.dates import ConciseDateFormatter
from matplotlib.typing import ColorType

from .encoding import RGB, Encoding, OKLab


def echogram(
    da_Sv: xr.DataArray,
    enc: Encoding,
    *,
    channel_idx: tuple[int, int, int] = (0, 1, 2),
    max_span: float | None = None,
    hwc: tuple[str, str, str] = ("depth", "ping_time", "channel"),
    ax: Axes | None = None,
    **plot_kwargs,
) -> QuadMesh:
    """
    Plot a trivariate echogram given a DataArray of volume backscattering strength.

    Parameters
    ----------
    da_Sv : xr.DataArray
        Sv data to plot.
    enc : Encoding
        Colour encoding, e.g. RGB() or OKLab().
    channel_idx : tuple[int, int, int], optional
        Indices of the reference, second and third channels in da_Sv.
    max_span : float, optional
        Maximum time span [s] between consecutive pings. Larger gaps are filled with NaN,
        and each quad is clipped to max_span / 2 on either side. By default no clipping.
    hwc : tuple[int, int, int], optional
        The name of the dimensions for height, width and channel in da_Sv.
    ax : Axes, optional
        Axes on which to draw the echogram. Uses the current Axes by default.
    **plot_kwargs
        Keyword arguments passed to plt.pcolormesh. `rasterized` defaults to True.

    Returns
    -------
    QuadMesh
        Echogram.
    """

    zdim, tdim, cdim = hwc

    # Keep the three channels, in (reference, f1, f2) order
    da_Sv = da_Sv.isel({cdim: list(channel_idx)}).transpose(*hwc)
    da_Sv.load()

    # Get data as numpy arrays, shape (depth, time, 3)
    sv: np.ndarray = da_Sv.values
    depth: np.ndarray = da_Sv[zdim].values
    time: np.ndarray = da_Sv[tdim].values

    # Compute RGB values
    rgb: np.ndarray = enc(sv)

    # Mask out invalid data (mostly bottom + some low insentity pixels)
    valid_data_mask: np.ndarray = ~np.all(np.isnan(sv), axis=-1)

    # Create echogram grid
    X, Y, gaps = _echogram_grid(time, depth, max_span)

    # Plot Quadmesh
    # Rasterize by default: one vector path per cell makes SVG export extremely slow
    plot_kwargs = {"rasterized": True} | plot_kwargs
    return _plot_mesh(ax, X, Y, rgb, valid_data_mask, gaps, plot_kwargs)


def colormap(
    enc: Encoding,
    max_Sv: float,
    lim: float,
    n=501,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    df1, df2 = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    # Reference Sv such that max(Sv_ref, Sv_f1, Sv_f2) = max_Sv
    ref = np.full_like(df1, max_Sv) - np.fmax(0, np.fmax(df1, df2))
    sv = np.stack([ref, ref + df1, ref + df2], axis=-1)
    rgb = enc(sv)
    return df1, df2, rgb


def plot_colormap(
    enc: Encoding,
    max_Sv: float,
    lim: float = 20,
    ax: Axes | None = None,
    samples: np.ndarray | None = None,
    paths_color: ColorType = "0.5",
):

    ax = ax or plt.gca()

    linekwargs = {
        "linestyle": "dashed",
        "linewidth": 0.5,
        "alpha": 0.4,
        "color": paths_color,
    }

    X, Y, C = colormap(enc, max_Sv, lim)
    ax.set_aspect(1)
    ax.pcolormesh(X, Y, C, rasterized=True)
    ax.axvline(0, **linekwargs)
    ax.axhline(0, **linekwargs)
    ax.axline((0, 0), slope=1, **linekwargs)
    ax.set_xlabel(r"$\Delta \text{MVBS}_{70 - 38 \text{ kHz}}$ [dB]")
    ax.set_ylabel(r"$\Delta \text{MVBS}_{120 - 38 \text{ kHz}}$ [dB]")
    ax.set_title(rf"$\max Sv = {max_Sv}$ dB")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Scale markers drawn above the colormap mesh and the reference lines
    linekwargs = {
        "linestyle": "dashed",
        "linewidth": 0.5,
        "color": "red",
        "label": "2D scale",
        "zorder": 10,
        "clip_on": False,
    }
    dotkwargs = {
        "facecolor": "none",
        "edgecolor": "red",
        "label": "1D scale",
        "zorder": 10,
        "clip_on": False,
    }
    if isinstance(enc, RGB):
        ax.axhline(-lim, **linekwargs)
        ax.scatter(-lim, -lim, **dotkwargs)
    elif isinstance(enc, OKLab):
        ax.axhline(0, **linekwargs)
        ax.scatter(0, 0, **dotkwargs)
    else:
        print(
            f"Warning: Unknown encodingn {type(enc)}, unable to plot 1D and 2D scales info."
        )

    if samples is not None:
        ax.scatter(
            samples[0],
            samples[1],
            s=3,
            linewidths=0.2,
            facecolors="none",
            edgecolors=paths_color,
            zorder=2,
        )


# ---- Helper functions ---- #


def _echogram_grid(
    time: np.ndarray,
    depth: np.ndarray,
    max_span: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell edges of the echogram mesh and the indices of the time gaps to fill.

    Returns
    -------
    X, Y : np.ndarray
        Meshgrid of time and depth cell edges, with one extra column per gap.
    gaps : np.ndarray
        Indices i such that time[i + 1] - time[i] > max_span.
    """
    # Time cell edges: midpoints between pings, with each half-width capped at max_span / 2.
    # Consecutive pings further apart than max_span get a masked column filling the gap.
    dt = np.diff(time)
    half_left = np.concatenate([dt[:1], dt]) / 2
    half_right = np.concatenate([dt, dt[-1:]]) / 2
    gaps = np.array([], dtype=int)
    if max_span is not None:
        span = np.timedelta64(round(max_span * 1000), "ms")
        half_left = np.minimum(half_left, span / 2)
        half_right = np.minimum(half_right, span / 2)
        gaps = np.flatnonzero(dt > span)

    time_edges = np.insert(
        np.concatenate([[time[0] - half_left[0]], time + half_right]),
        gaps + 2,
        time[gaps + 1] - half_left[gaps + 1],
    )

    # Depth cell edges: midpoints between samples, extrapolated at both ends
    depth_mid = (depth[1:] + depth[:-1]) / 2
    depth_edges = np.concatenate(
        [[2 * depth[0] - depth_mid[0]], depth_mid, [2 * depth[-1] - depth_mid[-1]]]
    )

    X, Y = np.meshgrid(time_edges, depth_edges)
    return X, Y, gaps


def _plot_mesh(
    ax: Axes | None,
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    valid_data_mask: np.ndarray,
    gaps: np.ndarray,
    plot_kwargs: dict,
) -> QuadMesh:
    """Fill gaps with masked columns, draw the mesh and format the time and depth axes.

    C is either a (depth, time) scalar array or a (depth, time, 3) RGB array.
    """
    C = np.insert(C, gaps + 1, 0.0, axis=1)
    valid_data_mask = np.insert(valid_data_mask, gaps + 1, False, axis=1)
    C = np.ma.array(C, mask=np.dstack([~valid_data_mask] * 3))

    ax = ax or plt.gca()
    im = ax.pcolormesh(X, Y, C, shading="flat", **plot_kwargs)

    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_inverted(True)

    return im
