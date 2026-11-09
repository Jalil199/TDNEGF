"""Shared input handling and checks for the pole-table generators.

Used by tools/generate_poles.py (MiniPole) and tools/fit_lorentzians.py.
A pole table approximates a real spectral function A(w) as

    A(w) ~ A_rec(w) = 2 Re sum_l R_l / (w - z_l),   Im z_l < 0,

and is written as two-column (real, imag) text files z_<name>_N<N>.txt and
R_<name>_N<N>.txt. TDNEGF multiplies the residues by 2*pi when it loads them
(load_poles_files / load_poles_square), so Gamma(w) = 2*pi * A_rec(w).

Stability: the lead self-energy built from the table is retarded (Herglotz)
only if A_rec(w) >= 0 for every real w. Where A_rec < 0 some modes of the
device Green's function grow slowly instead of decaying, which shows up only
at long times. check_table() tests this on a grid that resolves the
narrowest pole, and make_positive() repairs a table that fails it.
"""
import os

import numpy as np
from scipy.integrate import simpson


def semicircle_dos(w, t=1.0):
    """Normalized DOS of a 1D tight-binding lead (semicircle of half-width 2t)."""
    w = np.asarray(w, dtype=float)
    out = np.zeros_like(w)
    inside = np.abs(w) < 2 * t
    out[inside] = np.sqrt(4 * t**2 - w[inside] ** 2) / (2 * np.pi * t**2)
    return out


def bump_kernel(dw, eps):
    """Normalized C-infinity bump kernel with support [-eps, eps]."""
    half = int(np.ceil(eps / dw))
    x = np.arange(-half, half + 1) * dw
    k = np.zeros_like(x)
    inside = np.abs(x) < eps
    y = x[inside] / eps
    k[inside] = np.exp(1.0 / (y**2 - 1.0))
    return k / (k.sum() * dw)


def mollify(w, A, eps):
    """Smooth A(w) with a bump kernel of width eps, keeping its area.

    Sharp features (e.g. square-root band edges) make pole fits overshoot and
    produce A_rec < 0 next to them. Smoothing reduces the overshoot but does
    not guarantee A_rec >= 0 (the small fit error oscillates around zero
    wherever A vanishes); see make_positive().
    """
    if eps <= 0:
        return A
    area = simpson(A, x=w)
    As = np.convolve(A, bump_kernel(w[1] - w[0], eps), mode="same")
    area_s = simpson(As, x=w)
    return As * (area / area_s) if area_s > 0 else As


def reconstruct(w, z, R):
    """A_rec(w) = 2 Re sum_l R_l / (w - z_l)."""
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    for s in range(0, w.size, 20000):  # chunks keep memory bounded
        ws = w[s:s + 20000]
        out[s:s + 20000] = 2 * np.real(np.sum(R[None, :] / (ws[:, None] - z[None, :]), axis=1))
    return out


def check_grid(z, lo, hi):
    """Real-axis grid fine enough to resolve every pole, plus the far tails."""
    width = max(np.min(np.abs(z.imag)), 1e-6)
    a = min(lo, np.min(z.real)) - 10 * width
    b = max(hi, np.max(z.real)) + 10 * width
    n = int(min(4e6, max(2e4, 20 * (b - a) / width)))
    span = b - a
    tails = span * np.logspace(-2, 6, 600)
    return np.concatenate((a - tails[::-1], np.linspace(a, b, n), b + tails))


def check_table(z, R, lo, hi):
    """Return (min A_rec, argmin w, max Im z) on a grid that resolves all poles."""
    wc = check_grid(z, lo, hi)
    A = reconstruct(wc, z, R)
    i = np.argmin(A)
    return A[i], wc[i], np.max(z.imag)


def symmetry_center(w, A, rtol=1e-10):
    """Point c with A(c + x) = A(c - x), or None if A is not symmetric."""
    support = w[A > rtol * A.max()]
    c = 0.5 * (support[0] + support[-1])
    x = np.linspace(0, max(c - w[0], w[-1] - c), 20001)
    diff = np.interp(c + x, w, A, left=0, right=0) - np.interp(c - x, w, A, left=0, right=0)
    return c if np.max(np.abs(diff)) <= 1e-8 * A.max() else None


def make_positive(z, R, lo, hi, n_centers=81, n_widths=60, center=None):
    """Add one broad Lorentzian pole that makes A_rec >= 0 everywhere.

    The negative regions of a pole fit are tiny (|A_rec| ~ fit error) but
    spread over the band edges and tails. A single Lorentzian
    h g^2 / ((w - e)^2 + g^2) with the smallest height h that covers them all
    is chosen from a grid of centers e in [lo, hi] and widths g up to the
    largest |Im z_l| of the fit (so the time stepping is not made stiffer).
    It is positive everywhere and decays like 1/w^2, so it also dominates the
    far tails. The change to A is at most h (~ the fit error). If `center` is given (the symmetry point of
    a symmetric spectrum, see symmetry_center), e is fixed there so the
    correction keeps the symmetry. Returns (z, R, h, e, g); h = 0 if nothing
    was added.
    """
    wc = check_grid(z, lo, hi)
    A = reconstruct(wc, z, R)
    neg = A < 0
    if not neg.any():
        return z, R, 0.0, 0.0, 0.0
    wn, An = wc[neg], -A[neg]
    if wn.size > 20000:  # keep the deepest points and an even subsample
        keep = np.union1d(np.argsort(An)[-10000:], np.linspace(0, wn.size - 1, 10000).astype(int))
        wn, An = wn[keep], An[keep]
    best = None
    # Widths up to the widest existing pole: a wider one would add a faster
    # decaying mode and make the equations of motion stiffer.
    g_max = np.max(np.abs(z.imag))
    for g in np.logspace(np.log10(min(1e-2, g_max)), np.log10(g_max), n_widths):
        for e in (np.linspace(lo, hi, n_centers) if center is None else (center,)):
            h = np.max(An * ((wn - e) ** 2 + g**2)) / g**2
            if best is None or h < best[0]:
                best = (h, e, g)
    h, e, g = best
    h *= 1.01
    z, R = np.append(z, e - 1j * g), np.append(R, 0.5j * h * g)
    if check_table(z, R, lo, hi)[0] < 0:  # resolution/rounding limit: one more pass
        z, R, _, _, _ = make_positive(z, R, lo, hi, n_centers, n_widths, center)
    return z, R, h, e, g


def add_input_args(ap):
    """Arguments that select the spectral function (shared by both tools)."""
    ap.add_argument("kind", choices=["semicircle", "table", "expr"],
                    help="semicircle: 1D lead DOS; table: two-column file (w, A(w)); "
                         "expr: numpy expression in w, e.g. \"np.exp(-w**2)/np.sqrt(np.pi)\"")
    ap.add_argument("source", nargs="?",
                    help="file for 'table', expression for 'expr'")
    ap.add_argument("--t", type=float, default=1.0, help="lead hopping for 'semicircle' (default 1)")
    ap.add_argument("--wmin", type=float, help="lower end of the frequency grid (default: -10t, or the table range)")
    ap.add_argument("--wmax", type=float, help="upper end of the frequency grid (default: 10t, or the table range)")
    ap.add_argument("--n-grid", type=int, default=800001, help="grid points (default 800001)")
    ap.add_argument("--normalize", action="store_true",
                    help="rescale A(w) to unit area (default: keep the input area)")
    ap.add_argument("--name", help="label for the output files")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


def load_spectrum(args, ap):
    """Build (w, A, default_name) on a uniform grid from the parsed arguments."""
    if args.kind == "semicircle":
        lo = args.wmin if args.wmin is not None else -10 * args.t
        hi = args.wmax if args.wmax is not None else 10 * args.t
        w = np.linspace(lo, hi, args.n_grid)
        A, name = semicircle_dos(w, args.t), "Semicircle"
    elif args.kind == "table":
        if not args.source:
            ap.error("'table' needs a file: table my_dos.txt")
        data = np.loadtxt(args.source)
        w_in, A_in = data[:, 0], data[:, 1]
        order = np.argsort(w_in)
        w_in, A_in = w_in[order], A_in[order]
        lo = args.wmin if args.wmin is not None else w_in[0]
        hi = args.wmax if args.wmax is not None else w_in[-1]
        w = np.linspace(lo, hi, args.n_grid)
        A = np.interp(w, w_in, A_in, left=0.0, right=0.0)
        name = os.path.splitext(os.path.basename(args.source))[0]
    else:
        if not args.source:
            ap.error("'expr' needs an expression: expr \"np.exp(-w**2)/np.sqrt(np.pi)\"")
        if args.wmin is None or args.wmax is None:
            ap.error("'expr' needs --wmin and --wmax")
        w = np.linspace(args.wmin, args.wmax, args.n_grid)
        A = np.broadcast_to(eval(args.source, {"np": np, "w": w}), w.shape).astype(float)
        name = "Custom"
    if np.any(A < 0):
        ap.error(f"the input spectral function is negative (min {A.min():.2e}); it must be >= 0")
    if args.normalize:
        A = A / simpson(A, x=w)
    return w, A, name


def write_table(out_dir, name, z, R):
    os.makedirs(out_dir, exist_ok=True)
    N = z.size
    z_path = os.path.join(out_dir, f"z_{name}_N{N}.txt")
    R_path = os.path.join(out_dir, f"R_{name}_N{N}.txt")
    np.savetxt(z_path, np.column_stack((z.real, z.imag)))
    np.savetxt(R_path, np.column_stack((R.real, R.imag)))
    return z_path, R_path


def report(w, A_target, A_input, z, R, z_path, R_path):
    """Print accuracy and positivity of the table."""
    wc = np.linspace(w[0], w[-1], 20001)
    A_rec = reconstruct(wc, z, R)
    amin, wmin, imz = check_table(z, R, w[0], w[-1])
    print(f"{z.size} poles written to {z_path} and {R_path}")
    print(f"max |A_fit - A_rec|   = {np.max(np.abs(np.interp(wc, w, A_target) - A_rec)):.2e}"
          f"  (fit target; max A = {A_input.max():.3f})")
    if A_target is not A_input:
        print(f"max |A_input - A_rec| = {np.max(np.abs(np.interp(wc, w, A_input) - A_rec)):.2e}"
              "  (unsmoothed input)")
    print(f"min A_rec = {amin:+.2e} at w = {wmin:+.4f};  max Im z = {imz:+.2e}")
    if amin < 0 or imz >= 0:
        print("WARNING: A_rec < 0 somewhere (or a pole is not in the lower half plane): "
              "the self-energy is not retarded and long-time dynamics can grow.")
    return amin
