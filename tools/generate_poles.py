#!/usr/bin/env python3
"""Pole table for a lead spectral function with the minimal pole method.

Fits A(w) ~ 2 Re sum_l R_l / (w - z_l) with MiniPoleRf (Zhang, Erpenbeck, Yu &
Gull, J. Chem. Phys. 2025) and writes z_<name>_N<N>.txt / R_<name>_N<N>.txt
in the format of data/z_Semicircle_N49.txt. Requires MiniPole:

    pip install git+https://github.com/Green-Phys/MiniPole.git kneed

The complex residues give very accurate fits with few poles, but nothing in
the fit forces A_rec(w) >= 0: the fit error (~1e-8 to 1e-10) oscillates
around zero wherever A vanishes (band edges, tails). A negative region makes
the lead self-energy non-retarded and lets some modes grow slowly, which
shows up at long times. Two steps deal with this:

  1. Sharp features are smoothed with a bump kernel of width --eps; this
     improves the fit near band edges but does not by itself remove the
     negative regions (larger eps only moves them outward).
  2. If A_rec < 0 anywhere on a grid that resolves every pole, one broad
     positive Lorentzian is added (one extra pole, height ~ the fit error),
     which makes A_rec >= 0 everywhere. If the input is symmetric about some
     point, the Lorentzian is centered there and the symmetry is kept.
     Disable with --no-repair.

For a fit that is positive by construction (sum of Lorentzians with
non-negative weights, less accurate per pole) see tools/fit_lorentzians.py.

Examples
--------
Semicircle (1D lead, hopping t = 1) with at most 49 poles:
    python tools/generate_poles.py semicircle --M 49

Your own spectral function from a two-column file (w, A(w)):
    python tools/generate_poles.py table my_dos.txt --M 40 --name MyLead

Or from a numpy expression in w:
    python tools/generate_poles.py expr "np.exp(-w**2)/np.sqrt(np.pi)" --wmin -8 --wmax 8 --M 20

Load the result in Julia:
    Rλ, zλ = load_poles_files("data/z_MyLead_N40.txt", "data/R_MyLead_N40.txt")
"""
import argparse
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectra import add_input_args, check_table, load_spectrum, make_positive, mollify, symmetry_center, report, write_table  # noqa: E402


def fit_minipole(w, A, M, err, wp_max, k_max):
    if not hasattr(np, "complex_"):  # older MiniPole releases use np.complex_
        np.complex_ = np.complex128
    from mini_pole import MiniPoleRf

    A_fun = interp1d(w, A, kind="linear", bounds_error=False, fill_value=0.0)
    p = MiniPoleRf([A_fun], interval_type="infinite", wp_max=wp_max, err=err, M=M, k_max=k_max)
    return np.asarray(p.pole_location).ravel(), np.asarray(p.pole_weight)[:, 0, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--M", type=int, help="maximum number of poles (default: minimal number for --err)")
    ap.add_argument("--err", type=float, default=1e-11,
                    help="MiniPole error tolerance; sets the integration accuracy and, "
                         "without --M, the number of poles (default 1e-11)")
    ap.add_argument("--eps", type=float, default=0.1,
                    help="smoothing width, in units of t for 'semicircle' (default 0.1; 0 = no smoothing)")
    ap.add_argument("--no-repair", action="store_true",
                    help="do not add the positivity-restoring Lorentzian (the table may give A_rec < 0)")
    ap.add_argument("--wp-max", type=float, default=3.0, help="MiniPole Mobius-transform parameter (default 3)")
    ap.add_argument("--k-max", type=int, default=2000, help="MiniPole maximum number of contour integrals (default 2000)")
    args = ap.parse_args()

    w, A, name = load_spectrum(args, ap)
    scale = args.t if args.kind == "semicircle" else 1.0

    A_fit = mollify(w, A, args.eps * scale)
    z, R = fit_minipole(w, A_fit, args.M, args.err, args.wp_max, args.k_max)
    amin, wmin, _ = check_table(z, R, w[0], w[-1])
    print(f"MiniPole: {z.size} poles, min A_rec = {amin:+.2e} at w = {wmin:+.4f}")
    if amin < 0 and not args.no_repair:
        c = symmetry_center(w, A)
        if c is not None:
            print(f"input is symmetric about w = {c:+.4g}; the correction is centered there")
        z, R, h, e, g = make_positive(z, R, w[0], w[-1], center=c)
        print(f"added 1 positive Lorentzian (center {e:+.3f}, width {g:.3g}, height {h:.2e}) "
              "so that A_rec >= 0 everywhere")

    z_path, R_path = write_table(args.out_dir, args.name or name, z, R)
    report(w, A_fit, A, z, R, z_path, R_path)


if __name__ == "__main__":
    main()
