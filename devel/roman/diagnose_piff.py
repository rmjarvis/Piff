#!/usr/bin/env python3

"""Quick diagnostics for a fitted Piff file.

Usage:
    python devel/roman/diagnose_piff.py devel/roman/ffov_13906_1.piff
"""

import argparse
from collections import defaultdict

import numpy as np
import piff


def _get_sca(star):
    props = star.data.properties
    if "sca" in props:
        return int(props["sca"])
    if "chipnum" in props:
        return int(props["chipnum"])
    return -1


def _fmt(v):
    if np.isscalar(v):
        return f"{v:.4g}"
    return str(v)


def _collect_roman_sca_means(psf):
    if not (hasattr(psf, "components") and len(psf.components) > 0):
        return None, None
    comp0 = psf.components[0]
    if getattr(comp0, "_type_name", None) != "RomanOptics":
        return None, None

    model_num = comp0.model._num
    params_by_sca = defaultdict(list)
    for star in psf.stars:
        if star.is_flagged or star.is_reserve:
            continue
        params_by_sca[_get_sca(star)].append(star.fit.get_params(model_num))

    if not params_by_sca:
        return comp0, None

    sca_mean = {sca: np.mean(vals, axis=0) for sca, vals in params_by_sca.items()}
    return comp0, sca_mean


def _print_roman_aberration_diagnostics(sca_mean, focus_sca):
    if not sca_mean:
        return

    scas = np.array(sorted(sca_mean), dtype=int)
    params = np.array([sca_mean[s] for s in scas], dtype=float)
    nterms = params.shape[1]

    med = np.median(params, axis=0)
    mad = np.median(np.abs(params - med), axis=0)
    robust_sigma = 1.4826 * mad
    robust_sigma_safe = np.where(robust_sigma > 0, robust_sigma, np.nan)

    print("\nRomanOptics per-SCA aberration summary:")
    print("  (term 0 -> z4, term 1 -> z5, etc.)")
    for sca, p in zip(scas, params):
        max_i = int(np.argmax(np.abs(p)))
        l2 = np.linalg.norm(p)
        l1 = np.sum(np.abs(p))
        print(
            "  SCA {:2d}: ||p||2={:8.4f} ||p||1={:8.4f} max|term|=t{:02d}:{:8.4f}".format(
                int(sca), float(l2), float(l1), max_i, float(p[max_i])
            )
        )

    mean_abs = np.mean(np.abs(params), axis=0)
    print("\nRomanOptics per-term amplitudes across SCAs:")
    for j in range(nterms):
        print(
            "  term {:2d} (z{:2d}): mean|a|={:8.4f} med={:8.4f} robust_sigma={:8.4f}".format(
                j, j + 4, float(mean_abs[j]), float(med[j]),
                float(robust_sigma[j]) if np.isfinite(robust_sigma[j]) else np.nan
            )
        )

    if focus_sca is None:
        return
    elif focus_sca not in sca_mean:
        print(f"\nFocused SCA {focus_sca} not present in fitted usable stars.")
        return

    p = sca_mean[focus_sca]
    zscore = (p - med) / robust_sigma_safe
    print(f"\nFocused SCA {focus_sca} term-by-term offsets from focal-plane median:")
    for j, (val, m, dz, rs) in enumerate(zip(p, med, zscore, robust_sigma)):
        ztxt = f"{dz:+7.2f}" if np.isfinite(dz) else "   n/a "
        rstxt = f"{rs:8.4f}" if np.isfinite(rs) else "   n/a "
        print(
            "  term {:2d} (z{:2d}): val={:8.4f} med={:8.4f} delta={:+8.4f}"
            " robust_sigma={} robust_z={}".format(
                j, j + 4, float(val), float(m), float(val - m), rstxt, ztxt
            )
        )

    sca_norm = {int(s): float(np.linalg.norm(v)) for s, v in sca_mean.items()}
    worst = sorted(sca_norm.items(), key=lambda kv: -kv[1])[:5]
    print("\nTop 5 SCAs by RomanOptics ||p||2:")
    for sca, norm in worst:
        tag = "  <-- focused" if sca == focus_sca else ""
        print(f"  SCA {sca:2d}: {norm:8.4f}{tag}")


def summarize(psf, center_thresh, top, focus_sca):
    stars = psf.stars
    print(f"File summary: nstars={len(stars)}")
    if getattr(psf, "dof", 0):
        print(
            "  fit: chisq={:.3f} dof={} chisq/dof={:.6f} niter={} nremoved={}".format(
                float(psf.chisq),
                int(psf.dof),
                float(psf.chisq) / float(psf.dof),
                int(getattr(psf, "niter", 0)),
                int(getattr(psf, "nremoved", 0)),
            )
        )
    else:
        print("  fit metadata dof is 0 or unavailable")

    rows = []
    for i, star in enumerate(stars):
        chisq = np.nan if star.fit.chisq is None else float(star.fit.chisq)
        dof = -1 if star.fit.dof is None else int(star.fit.dof)
        flux = np.nan if star.fit.flux is None else float(star.fit.flux)
        cx, cy = star.fit.center
        rows.append(
            (
                i,
                _get_sca(star),
                chisq,
                dof,
                flux,
                float(cx),
                float(cy),
                bool(star.is_flagged),
                bool(star.is_reserve),
                float(star.x),
                float(star.y),
            )
        )
    arr = np.array(
        rows,
        dtype=[
            ("idx", "i4"),
            ("sca", "i4"),
            ("chisq", "f8"),
            ("dof", "i4"),
            ("flux", "f8"),
            ("cx", "f8"),
            ("cy", "f8"),
            ("flagged", "?"),
            ("reserve", "?"),
            ("x", "f8"),
            ("y", "f8"),
        ],
    )
    good = (~arr["flagged"]) & (~arr["reserve"]) & (arr["dof"] > 0)
    print(f"  usable stars (not flagged/reserve): {int(np.sum(good))}")

    if np.any(good):
        chi = arr["chisq"][good] / arr["dof"][good]
        print(
            "  usable chi2/dof: median={:.4f} p90={:.4f} max={:.4f}".format(
                float(np.median(chi)),
                float(np.percentile(chi, 90)),
                float(np.max(chi)),
            )
        )

    center = np.hypot(arr["cx"], arr["cy"])
    large_center = center > center_thresh
    print(
        f"  stars with |center| > {center_thresh:g}: {int(np.sum(large_center))}"
        f" ({int(np.sum(large_center & good))} usable)"
    )
    print(f"  stars with negative flux: {int(np.sum(arr['flux'] < 0))}")

    print("\nPer-SCA (usable stars):")
    for sca in sorted(np.unique(arr["sca"])):
        m = good & (arr["sca"] == sca)
        if not np.any(m):
            continue
        c = arr["chisq"][m] / arr["dof"][m]
        cen = np.hypot(arr["cx"][m], arr["cy"][m])
        print(
            f"  SCA {sca:2d}: n={int(np.sum(m)):3d}"
            f" med(chi2/dof)={np.median(c):7.3f}"
            f" p90={np.percentile(c, 90):7.3f}"
            f" max={np.max(c):7.3f}"
            f" med|cen|={np.median(cen):6.3f}"
            f" max|cen|={np.max(cen):6.3f}"
        )

    print(f"\nWorst {top} stars by chi2/dof:")
    denom = np.maximum(arr["dof"], 1)
    order = np.argsort(-(arr["chisq"] / denom))
    for j in order[:top]:
        cdr = arr["chisq"][j] / max(arr["dof"][j], 1)
        cen = np.hypot(arr["cx"][j], arr["cy"][j])
        print(
            "  idx={} sca={} chi2/dof={:.3f} |cen|={:.3f} flux={} flagged={} reserve={}"
            " x={:.1f} y={:.1f}".format(
                int(arr["idx"][j]),
                int(arr["sca"][j]),
                float(cdr),
                float(cen),
                _fmt(arr["flux"][j]),
                bool(arr["flagged"][j]),
                bool(arr["reserve"][j]),
                float(arr["x"][j]),
                float(arr["y"][j]),
            )
        )

    comp0, sca_mean = _collect_roman_sca_means(psf)
    if comp0 is not None and sca_mean:
        print("\nComponent 0 (RomanOptics) mean params by SCA:")
        for sca in sorted(sca_mean):
            pstr = np.array2string(
                sca_mean[sca],
                formatter={"float_kind": lambda x: f"{x:.4f}"},
            )
            print(f"  SCA {sca:2d}: {pstr}")
        _print_roman_aberration_diagnostics(sca_mean, focus_sca=focus_sca)


def main():
    parser = argparse.ArgumentParser(description="Summarize diagnostics from a .piff file")
    parser.add_argument("piff_file", help="Path to .piff file")
    parser.add_argument(
        "--center-thresh",
        type=float,
        default=0.5,
        help="Threshold in pixels for reporting large fitted centers",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many worst stars (by chi2/dof) to print",
    )
    parser.add_argument(
        "--focus-sca",
        type=int,
        default=None,
        help="SCA to print detailed aberration diagnostics for",
    )
    args = parser.parse_args()

    psf = piff.read(args.piff_file)
    summarize(
        psf,
        center_thresh=args.center_thresh,
        top=args.top,
        focus_sca=args.focus_sca,
    )


if __name__ == "__main__":
    main()
