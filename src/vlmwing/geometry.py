"""Wing planform, airfoil camber and spanwise discretization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Wing", "Mesh", "naca4_camber_slope", "BASELINE"]


def naca4_camber_slope(naca: str, x_over_c: float = 0.75) -> float:
    """Slope dz/dx of a NACA 4-digit mean camber line at ``x_over_c``.

    Parameters
    ----------
    naca:
        Four-digit designation, e.g. ``"3217"``.
    x_over_c:
        Chordwise station in fractions of chord (0.75 is the VLM control point).
    """
    if len(naca) != 4 or not naca.isdigit():
        raise ValueError(f"expected a 4-digit NACA designation, got {naca!r}")
    m = int(naca[0]) / 100.0
    p = int(naca[1]) / 10.0
    if m == 0.0:
        return 0.0
    if p == 0.0:
        raise ValueError(f"cambered NACA {naca} needs a non-zero camber position")
    if x_over_c < p:
        return 2.0 * m / p**2 * (p - x_over_c)
    return 2.0 * m / (1.0 - p) ** 2 * (p - x_over_c)


@dataclass(frozen=True)
class Wing:
    """Straight-tapered, swept wing with linear geometric twist.

    Lengths in metres, angles in degrees. Twist varies linearly from 0 at the
    root to ``tip_twist_deg`` at the tip (negative is washout).
    ``reference_area`` is used to normalize coefficients; it defaults to the
    planform area.
    """

    span: float
    root_chord: float
    tip_chord: float
    sweep_c4_deg: float
    tip_twist_deg: float
    airfoil: str = "3217"
    reference_area: float | None = None

    def __post_init__(self) -> None:
        if self.span <= 0 or self.root_chord <= 0 or self.tip_chord <= 0:
            raise ValueError("span and chords must be positive")
        naca4_camber_slope(self.airfoil)  # validates the designation

    # ---- construction -------------------------------------------------
    @classmethod
    def from_planform(
        cls,
        area: float,
        aspect_ratio: float,
        taper: float,
        sweep_c4_deg: float,
        tip_twist_deg: float,
        airfoil: str = "3217",
    ) -> Wing:
        """Build a wing from area, aspect ratio and taper ratio."""
        span = float(np.sqrt(aspect_ratio * area))
        root = 2.0 * area / (span * (1.0 + taper))
        return cls(span, root, taper * root, sweep_c4_deg, tip_twist_deg, airfoil)

    def with_(self, **changes) -> Wing:
        """Return a copy with some fields changed."""
        return replace(self, **changes)

    # ---- derived geometry ---------------------------------------------
    @property
    def semispan(self) -> float:
        return self.span / 2.0

    @property
    def taper(self) -> float:
        return self.tip_chord / self.root_chord

    @property
    def planform_area(self) -> float:
        return self.span * (self.root_chord + self.tip_chord) / 2.0

    @property
    def area(self) -> float:
        """Reference area used for all coefficients."""
        return self.planform_area if self.reference_area is None else self.reference_area

    @property
    def aspect_ratio(self) -> float:
        return self.span**2 / self.planform_area

    @property
    def tan_sweep_le(self) -> float:
        return (
            np.tan(np.radians(self.sweep_c4_deg)) + 0.25 * (self.root_chord - self.tip_chord) / self.semispan
        )

    @property
    def sweep_le_deg(self) -> float:
        return float(np.degrees(np.arctan(self.tan_sweep_le)))

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord."""
        lam = self.taper
        return 2.0 / 3.0 * self.root_chord * (1 + lam + lam**2) / (1 + lam)

    @property
    def y_ac(self) -> float:
        """Spanwise station of the mean aerodynamic chord (semispan)."""
        lam = self.taper
        return self.span / 6.0 * (1 + 2 * lam) / (1 + lam)

    @property
    def x_ac(self) -> float:
        """Chordwise position of the wing aerodynamic center (quarter MAC)."""
        return self.y_ac * self.tan_sweep_le + 0.25 * self.mac

    @property
    def camber_slope(self) -> float:
        """Camber-line slope at the 3/4-chord control point."""
        return naca4_camber_slope(self.airfoil, 0.75)

    def chord(self, y: np.ndarray) -> np.ndarray:
        return self.root_chord - (self.root_chord - self.tip_chord) * np.abs(y) / self.semispan

    def x_le(self, y: np.ndarray) -> np.ndarray:
        return np.abs(y) * self.tan_sweep_le

    def twist_rad(self, y: np.ndarray) -> np.ndarray:
        return np.radians(self.tip_twist_deg * np.abs(y) / self.semispan)

    def mesh(self, n_panels: int) -> Mesh:
        """Uniform spanwise mesh with one chordwise panel."""
        return Mesh.build(self, n_panels)


@dataclass(frozen=True)
class Mesh:
    """Horseshoe-vortex lattice: one chordwise panel, ``n`` spanwise panels."""

    y_nodes: np.ndarray  # panel edges, shape (n+1,)
    y_mid: np.ndarray  # control-point stations, shape (n,)
    chord_mid: np.ndarray
    x_c4_nodes: np.ndarray  # bound-vortex end points (quarter chord)
    x_c4_mid: np.ndarray
    x_ctrl: np.ndarray  # control points (three-quarter chord)
    dy: np.ndarray  # panel widths

    @property
    def n(self) -> int:
        return self.y_mid.size

    @classmethod
    def build(cls, wing: Wing, n_panels: int) -> Mesh:
        if n_panels < 2 or n_panels % 2:
            raise ValueError("n_panels must be an even integer >= 2 (whole wing)")
        y_nodes = np.linspace(-wing.semispan, wing.semispan, n_panels + 1)
        y_mid = 0.5 * (y_nodes[:-1] + y_nodes[1:])
        chord_nodes = wing.chord(y_nodes)
        chord_mid = wing.chord(y_mid)
        return cls(
            y_nodes=y_nodes,
            y_mid=y_mid,
            chord_mid=chord_mid,
            x_c4_nodes=wing.x_le(y_nodes) + 0.25 * chord_nodes,
            x_c4_mid=wing.x_le(y_mid) + 0.25 * chord_mid,
            x_ctrl=wing.x_le(y_mid) + 0.75 * chord_mid,
            dy=np.diff(y_nodes),
        )


#: Baseline wing of the study (W=9, X=3, Y=5, Z=5): S = 1 m², A = 15,
#: lambda = 0.65, Lambda_c/4 = 12 deg, tip twist -5 deg, NACA 3217.
#: Span and chords are the rounded values used in the report.
BASELINE = Wing(
    span=3.873,
    root_chord=0.313,
    tip_chord=0.203,
    sweep_c4_deg=12.0,
    tip_twist_deg=-5.0,
    airfoil="3217",
    reference_area=1.0,
)
