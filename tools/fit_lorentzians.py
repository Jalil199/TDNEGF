#!/usr/bin/env python3
"""Pole table for a lead spectral function as a sum of Lorentzians.

Fits

    A(w) ~ sum_k c_k w_k^2 / ((w - e_k)^2 + w_k^2),   c_k >= 0,

with e_k, w_k from nonlinear least squares and c_k from NNLS (variable
projection), the approach of the legacy data/z_Lorentz_N31.txt table. Each
Lorentzian is one pole z = e - i w with residue R = i c w / 2, written in the
same format as tools/generate_poles.py.

Because every term is a positive Lorentzian, A_rec(w) > 0 for all w and the
lead self-energy is retarded by construction: no growing modes at any time.
The price is accuracy per pole; tools/generate_poles.py (MiniPole) needs far
fewer poles for the same error. Needs only numpy/scipy.

Examples
--------
Semicircle with 31 Lorentzians:
    python tools/fit_lorentzians.py semicircle --K 31

Your own spectral function:
    python tools/fit_lorentzians.py table my_dos.txt --K 40 --name MyLead
    python tools/fit_lorentzians.py expr "np.exp(-w**2)/np.sqrt(np.pi)" --wmin -8 --wmax 8 --K 10
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares, nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectra import add_input_args, load_spectrum, mollify, report, write_table  # noqa: E402


def fit_lorentz(w, A, K, n_fit=4000, max_nfev=2000):
    """Positions/widths by least squares, weights by NNLS. Returns (z, R)."""
    support = w[A > 1e-6 * A.max()]
    lo, hi = support.min(), support.max()
    margin = 0.25 * (hi - lo)
    wf = np.linspace(lo - margin, hi + margin, n_fit)
    y = np.interp(wf, w, A)

    def basis(theta):
        e, ww = theta[:K], np.exp(theta[K:])
        return ww[None, :] ** 2 / ((wf[:, None] - e[None, :]) ** 2 + ww[None, :] ** 2)

    def residual(theta):
        Phi = basis(theta)
        return Phi @ nnls(Phi, y)[0] - y

    e0 = np.linspace(lo, hi, K)
    w0 = np.full(K, (hi - lo) / K)
    sol = least_squares(residual, np.concatenate((e0, np.log(w0))), max_nfev=max_nfev)
    c = nnls(basis(sol.x), y)[0]
    e, ww = sol.x[:K], np.exp(sol.x[K:])
    keep = c > 0
    return (e - 1j * ww)[keep], (0.5j * c * ww)[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--K", type=int, required=True, help="number of Lorentzians")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="optional smoothing width, in units of t for 'semicircle' (default 0)")
    ap.add_argument("--max-nfev", type=int, default=2000, help="least-squares iterations (default 2000)")
    args = ap.parse_args()

    w, A, name = load_spectrum(args, ap)
    scale = args.t if args.kind == "semicircle" else 1.0
    A_fit = mollify(w, A, args.eps * scale)
    z, R = fit_lorentz(w, A_fit, args.K, max_nfev=args.max_nfev)
    if z.size < args.K:
        print(f"note: {args.K - z.size} Lorentzians got zero weight and were dropped")
    z_path, R_path = write_table(args.out_dir, args.name or name + "Lorentz", z, R)
    report(w, A_fit, A, z, R, z_path, R_path)


if __name__ == "__main__":
    main()
