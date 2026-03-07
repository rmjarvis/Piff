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


def summarize(psf, center_thresh, top):
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

    if hasattr(psf, "components") and len(psf.components) > 0:
        comp0 = psf.components[0]
        if getattr(comp0, "_type_name", None) == "RomanOptics":
            model_num = comp0.model._num
            params_by_sca = defaultdict(list)
            for star in stars:
                if star.is_flagged or star.is_reserve:
                    continue
                params_by_sca[_get_sca(star)].append(star.fit.get_params(model_num))
            if params_by_sca:
                print("\nComponent 0 (RomanOptics) mean params by SCA:")
                for sca in sorted(params_by_sca):
                    p = np.mean(params_by_sca[sca], axis=0)
                    pstr = np.array2string(
                        p,
                        formatter={"float_kind": lambda x: f"{x:.4f}"},
                    )
                    print(f"  SCA {sca:2d}: {pstr}")


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
    args = parser.parse_args()

    psf = piff.read(args.piff_file)
    summarize(psf, center_thresh=args.center_thresh, top=args.top)


if __name__ == "__main__":
    main()
