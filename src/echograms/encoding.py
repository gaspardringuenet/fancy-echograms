from dataclasses import dataclass
from typing import Protocol

import numpy as np

# OKLab to sRGB conversion matrices (https://bottosson.github.io/posts/oklab/)
_LAB_TO_LMS = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


class Encoding(Protocol):
    def __call__(self, sv: np.ndarray) -> np.ndarray:  # type: ignore
        """Sv [dB] of (reference, f1, f2) along the last axis, shape (..., 3), NaN if missing
        -> RGB (..., 3) in [0, 1]."""


def _max_and_deltas(sv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """max over channels (NaN ignored), and ΔSv(f1), ΔSv(f2) relative to the reference."""
    max_Sv = np.fmax.reduce(sv, axis=-1)
    ref = sv[..., 0]
    return max_Sv, sv[..., 1] - ref, sv[..., 2] - ref


@dataclass(frozen=True)
class RGB:
    """Encoding of 3-channel Sv data using a simple RGB mapping: Sv(f0) -> R, Sv(f1) -> G,
    Sv(f3) -> B.

    Parameters
    ----------
    vmin : float, default -90.0
        Minimum Sv [dB] bound of the colormap.
    vmax : float, default -50.0
        Maximum Sv [dB] bound of the colormap.
    """

    vmin: float = -90.0
    vmax: float = -50.0

    def __call__(self, sv: np.ndarray) -> np.ndarray:

        if not sv.shape[-1] == 3:
            raise ValueError(
                f"Invalid Sv array shape {sv.shape}. Expected HWC with C==3."
            )

        rgb = ((sv - self.vmin) / (self.vmax - self.vmin)).clip(0, 1)
        # A missing channel is rendered as 0 (dark), not as equal to the reference
        return np.nan_to_num(rgb, nan=0.0)


@dataclass(frozen=True)
class OKLab:
    """Perceptual encoding in OKLab: max Sv sets lightness, ΔSv's set hue and chroma.

    Default settings were manually tuned to align with an RGB encoding as much as possible,
    as well as avoid chroma aberrance.

    Parameters
    ----------
    vmin, vmax : float, default -90, -50
        Sv [dB] mapped linearly to lightness, clipped outside this range.
    sat_max : float, default 25
        ΔSv [dB] at which chroma reaches c_max.
    l_range : tuple[float, float], default (0.25, 0.75)
        OKLab lightness at vmin and vmax.
    c_max : float, default 0.15
        Maximum OKLab chroma, reached at a ΔSv = sat_max.
    theta : float, default 0.85 pi
        Rotation [rad] applied to the (ΔSv(f1), ΔSv(f2)) direction to get the (a, b) hue.
    """

    vmin: float = -90
    vmax: float = -50
    sat_max: float = 25
    l_range: tuple[float, float] = (0.25, 0.75)
    c_max: float = 0.15
    theta: float = 0.85 * np.pi

    def __call__(self, sv: np.ndarray) -> np.ndarray:

        if not sv.shape[-1] == 3:
            raise ValueError(
                f"Invalid Sv array shape {sv.shape}. Expected HWC with C==3."
            )

        max_Sv, df1, df2 = _max_and_deltas(sv)
        # NaN (empty pixels, masked at plot time) must not reach the gamut check: NaN fails every
        # comparison, so those pixels would all go through the costly chroma bisection
        value: np.ndarray = np.nan_to_num(
            ((max_Sv - self.vmin) / (self.vmax - self.vmin)).clip(0, 1),
            nan=0.0,
        )
        # Missing channel deltas are neutral (grey)
        return _oklab_rgb(
            value,
            np.nan_to_num(df1),
            np.nan_to_num(df2),
            sat_max=self.sat_max,
            l_range=self.l_range,
            c_max=self.c_max,
            theta=self.theta,
        )


# --- Helper functions --- #


def _oklab_rgb(
    value: np.ndarray,
    df1: np.ndarray,
    df2: np.ndarray,
    sat_max: float = 25.0,
    l_range: tuple[float, float] = (0.25, 0.75),
    c_max: float = 0.15,
    theta: float = 0.85 * np.pi,
) -> np.ndarray:
    """Convert OKLab coordinates to sRGB, with gamut clipping by chroma reduction.

    Lightness is linear in value. (a, b) is linear in (df1, df2): the chroma c_max is reached at
    a spectral difference of sat_max, then clipped radially. Colours outside the sRGB gamut have
    their chroma reduced at constant lightness and hue, as oklab.clamp does.

    Parameters
    ----------
    max_Sv : np.ndarray
        max(Sv) accross channel, normalised in [0, 1], mapped linearly to lightness over l_range.
    df1, df2 : np.ndarray
        ΔSv(f1), ΔSv(f2) [dB], same shape as value. Must not contain NaN.
    sat_max, l_range, c_max, theta
        See OKLab.

    Returns
    -------
    np.ndarray
        sRGB values in [0, 1], shape (*value.shape, 3).
    """

    lightness = l_range[0] + (l_range[1] - l_range[0]) * value
    r = np.hypot(df1, df2)
    chroma = c_max * np.minimum(r / sat_max, 1.0)

    # Rotate the unit direction (df1, df2) / r by theta without trigonometric calls per pixel
    unit = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
    cos_a = unit * (df1 * np.cos(theta) - df2 * np.sin(theta))
    sin_a = unit * (df1 * np.sin(theta) + df2 * np.cos(theta))

    ab = (chroma * np.stack([cos_a, sin_a])).reshape(2, -1)
    light = lightness.ravel()

    def to_srgb_linear(idx, scale):
        # scale multiplies chroma only, so bisection can shrink it while keeping L and hue
        lab = np.stack([light[idx], scale * ab[0, idx], scale * ab[1, idx]], axis=-1)
        return ((lab @ _LAB_TO_LMS.T) ** 3) @ _LMS_TO_RGB.T

    def in_gamut(lin):
        # Small tolerance so that floating-point noise at the gamut boundary is not reduced
        return np.all((lin >= -1e-6) & (lin <= 1 + 1e-6), axis=-1)

    # Full chroma first; only out-of-gamut pixels go through the bisection below
    lin = to_srgb_linear(slice(None), 1.0)

    # Chroma reduction by bisection, only for out-of-gamut pixels
    bad = np.flatnonzero(~in_gamut(lin))
    if bad.size:
        lo, hi = np.zeros(bad.size), np.ones(bad.size)
        for _ in range(16):
            mid = (lo + hi) / 2
            good = in_gamut(to_srgb_linear(bad, mid))
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        lin[bad] = to_srgb_linear(bad, lo)

    lin = lin.clip(0, 1).reshape(*lightness.shape, 3)
    # sRGB transfer function (linear -> gamma-encoded)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
