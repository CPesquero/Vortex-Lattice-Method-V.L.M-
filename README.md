# Vortex Lattice Method (V.L.M.)

[![Python 3](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t-bse9iqq3N1KWBe4jeSjNakV9HRVNxZ?usp=sharing)

**3D aerodynamic characterization and stall optimization of a swept, tapered, twisted wing.**

A from-scratch Python implementation of the vortex lattice method. It builds the wing, solves the horseshoe-vortex system, and extracts lift, induced drag and pitching moment. It checks the result against XFLR5, then searches for a planform that makes the wing stall at the root first.

![Local lift distribution](figures/Figure%203.png)

The picture is the core of the project: the spanwise lift distribution at five angles of attack, including the central dip that sweep produces, which a lifting-line model does not capture.

| | measured |
|---|---|
| Lift-curve slope | C<sub>Lα</sub> = 0.093 per degree |
| Lift at zero incidence | C<sub>L,α=0</sub> = 0.068 (cambered NACA 3217) |
| Zero-lift angle | α<sub>L=0</sub> = −0.733° |
| Induced drag | C<sub>Di</sub> = 0.0007 at 0°, 0.0082 at 6°, 0.0288 at 12° |
| Moment about the aerodynamic center | C<sub>M,ac</sub> = 0.034 to 0.055 over −12° to 12° |
| Mesh convergence | C<sub>L</sub> varies by less than 0.5 % from 80 semispan panels onward |
| Optimized design | stall onset at 15 % of the semispan, dC<sub>l</sub>/dy = −0.201 |

Every number above is reproduced from the report in [`docs/`](docs/) (see below). The limits of what they mean are stated in [Limitations](#limitations), not hidden.

## The wing

The baseline airfoil is the NACA 3217 (3 % camber at 20 % chord, 17 % thickness). The planform follows from the assignment parameters W = 9, X = 3, Y = 5, Z = 5:

![Airfoil Representation](figures/Figure%201.png)

| Parameter | Value |
|---|---|
| Wing area S | 1 m² |
| Aspect ratio A | 15 |
| Taper ratio λ | 0.65 |
| Sweep at quarter chord Λ<sub>1/4</sub> | 12° |
| Geometric twist at the tip ε | −5° |
| Panels | 1 chordwise × 10 per semispan |

Derived geometry: span b = 3.873 m, root chord 0.313 m, tip chord 0.203 m, mean aerodynamic chord 0.262 m, wing aerodynamic center at (x, y) = (0.2694, 0.8994) m.

![Wing planform](figures/Figure%202.png)

## What it computes

The pipeline follows the structure of the report, one section per stage.

1. **Geometry.** Wing slope at x/c = 0.75 from Pistolesi's theorem, span, root, tip, mean and aerodynamic chords, and the aerodynamic center.
2. **Local lift.** The horseshoe-vortex influence matrix is solved with the no-penetration condition at the 3/4-chord control point of each panel, including twist and the camber-line slope. The Kutta–Joukowski theorem then gives the local lift coefficient.
3. **Global lift.** C<sub>L</sub> is integrated along the span. C<sub>Lα</sub>, C<sub>L,α=0</sub> and α<sub>L=0</sub> are taken from the resulting curve.
4. **Basic and additional lift.** The distribution is split into the part due to camber and twist (the basic load, solved at the zero-lift angle) and the part due to incidence, normalized to C<sub>L</sub> = 1.
5. **Induced drag.** Downwash is evaluated in the Trefftz plane from the wake vortices, which gives the induced angle at each station and the local and global C<sub>Di</sub>.
6. **Pitching moment.** Moments are taken about the origin and transferred to the aerodynamic center.
7. **Validation.** Local lift is compared against XFLR5, and a mesh-convergence study fixes the panel count.
8. **Stall optimization.** A parameter search finds planforms whose stall starts inboard.

```text
5 angles of attack  ->  influence matrix  ->  circulation  ->  C_l(y), C_L, C_Di, C_M
                                                    |
                              basic / additional split, XFLR5 check, mesh study
                                                    |
                         parameter search  ->  five closest feasible redesigns
```

## Results

### Lift

Local lift at each angle of attack (the table is in section 4 of the report) rises almost uniformly with α. The slope of the global curve is linear over −12° to 12°, which is expected in the attached-flow regime this method assumes.

![C_L versus alpha](figures/Figure%204.png)

### Basic and Additional Lift

Under the linear potential theory, the lift distribution is decoupled into two components: the basic lift due to camber and twist (solved at the zero-lift angle, $C_L = 0$), and the additional lift due to incidence (normalized to $C_L = 1$). This validates that the geometric twist correctly unloads the tips at cruise.

![Basic and additional lift](figures/Figure%205.png)

### Induced drag

Induced drag is smallest near zero incidence and grows with |α|. Its spanwise distribution concentrates toward the tips at negative angles of attack, where the twisted tip carries the larger load in the opposite direction.

![Local induced drag](figures/Figure%206.png)

### Pitching moment

The moment about the origin is strongly α-dependent, as expected. About the aerodynamic center it is small and almost flat, which is the signature of a correctly located aerodynamic center. See [Limitations](#limitations) for the small residual trend.

![Pitching moment](figures/Figure%207.png)

### Validation against XFLR5

The model reproduces the central dip in the lift distribution caused by the 12° sweep. XFLR5's default lifting-line analysis does not show it. The spacing between curves at different angles of attack agrees, which supports the lift slope.

![Comparison with XFLR5](figures/Figure%208.png)

### Mesh convergence

Between 4 and 120 semispan panels at α = 6° the lift coefficient moves from 0.639 to 0.619 while compute time grows from 0.5 ms to 0.10 s. From 80 panels onward the variation is under 0.5 %, so N<sub>y</sub> = 80 is the working configuration.

| N<sub>y</sub> | C<sub>L</sub> | Time [s] |
|---|---|---|
| 4 | 0.639 | 0.0005 |
| 10 | 0.634 | 0.0009 |
| 20 | 0.628 | 0.0031 |
| 40 | 0.623 | 0.0145 |
| 80 | 0.621 | 0.0467 |
| 120 | 0.619 | 0.1033 |

## Stall optimization

**Goal.** Redesign the wing so that the loss of lift starts at or inside 40 % of the semispan, with a spanwise lift gradient of at least dC<sub>l</sub>/dy = −0.2.

**Criterion.** Stall onset is taken as the station where the local lift demand C<sub>l</sub>(y) peaks. Under the hypothesis of a linear potential regime, it is assumed that the airfoil maintains a maximum capacity c<sub>l,max</sub> that is uniform along the wingspan. Consequently, the point where the lift peak is reached coincides with the critical section where the boundary layer will separate in real flight. This is a modeling assumption, not a measurement: the VLM itself does not see boundary-layer separation.

**Search.** The code iterates over aspect ratio (by changing the span), sweep, taper ratio and twist, keeps every combination that satisfies the goal, and ranks them by how little they depart from the baseline. It returns the five closest.

**Best design found.**

| | Baseline | Optimized |
|---|---|---|
| Airfoil | NACA 3217 | NACA 4417 |
| Aspect ratio | 15 | 7.24 |
| Sweep | 12° | 14.74° |
| Taper ratio | 0.65 | 0.905 |
| Tip twist | −5° | −9° |
| Stall onset | | 15 % of the semispan |
| dC<sub>l</sub>/dy | | −0.201 |

![Optimized lift distribution](figures/Figure%2010.png)

The trade-offs, as discussed in the report: more negative twist moves the lift peak inboard but can push the tips to negative lift in cruise; higher taper gives more chord to carry the load but moves the center of pressure outward and departs from the elliptical distribution; the lower aspect ratio costs induced drag; and a thicker root airfoil gives a gentler stall at the price of more drag.

## Limitations

These are worth knowing before relying on any figure above.

- **Linear, attached-flow model.** The C<sub>L</sub>–α curve is linear by construction. There is no viscous correction and no real stall prediction.
- **The stall criterion is a proxy.** It locates the lift peak, not boundary-layer separation, and assumes a spanwise-uniform c<sub>l,max</sub>. The airfoil change in the optimization (NACA 4417) is not seen by the VLM.
- **One chordwise panel.** The baseline uses N<sub>x</sub> = 1, which places the lift on the quarter-chord line.
- **XFLR5 agreement is qualitative.** Curve shapes and spacing agree. In the comparison figure the in-house curves and XFLR5 points are offset by a visible amount, so this is a consistency check, not a quantitative match.
- **C<sub>M,ac</sub> is nearly, not exactly, constant.** It drifts from 0.034 to 0.055 over the α range, so "independent of α" is an approximation.
- **Pitching moment is the wing alone.** No fuselage, tail or propulsion, so it says nothing about aircraft-level stability.

## Running it

<!-- TODO: replace with the real entry points of the repository -->

```bash
git clone https://github.com/CPesquero/vlm-wing-aerodynamics.git
cd vlm-wing-aerodynamics
pip install numpy scipy matplotlib
```

The full code is also available as a notebook in Google Colab: [Google Colab](https://colab.research.google.com/drive/1t-bse9iqq3N1KWBe4jeSjNakV9HRVNxZ?usp=sharing).

## Documentation

| Document | What it is |
|---|---|
| [`docs/vlm_aerodynamics_report.pdf`](docs/vlm_aerodynamics_report.pdf) | The full report: derivations, tables, figures and conclusions |

## References

1. F. Gandía, *Aerodinámica, Cap. VII: Alas en régimen incompresible*, Universidad Politécnica de Madrid.
2. A. Deperrois, *XFLR5*.

## Authors

Lucas García Franco and Carlos Pesquero Cañizares. Universidad Politécnica de Madrid, ETSIAE.
