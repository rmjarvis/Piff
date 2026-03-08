#!/usr/bin/env python3

"""Diagnose Roman bilinear-corner approximation error against direct getPSF.

Example:
    python devel/roman/diagnose_bilinear_vs_direct.py --grid 5 --plot devel/roman/bilinear_map.png
"""

import argparse
import os

import galsim
import numpy as np
import piff
import piff.roman_psf


def parse_scas(spec):
    if spec is None or spec.strip() == "":
        return list(range(1, 19))
    out = []
    for token in spec.split(","):
        tok = token.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    out = sorted(set(out))
    for sca in out:
        if sca < 1 or sca > 18:
            raise ValueError(f"Invalid SCA {sca}. Must be in 1..18.")
    return out


def parse_aberrations(spec, nparam):
    if spec is None:
        return np.zeros(nparam, dtype=float)
    vals = [float(v.strip()) for v in spec.split(",") if v.strip() != ""]
    if len(vals) != nparam:
        raise ValueError(f"--aberrations requires {nparam} values (got {len(vals)})")
    return np.array(vals, dtype=float)


def point_metrics(model_image, direct_image):
    diff = model_image - direct_image
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    peak = float(np.max(np.abs(direct_image)))
    l2 = float(np.sqrt(np.sum(direct_image**2)))
    frac_peak = np.nan if peak == 0.0 else max_abs / peak
    frac_l2 = np.nan if l2 == 0.0 else float(np.sqrt(np.sum(diff**2)) / l2)
    return max_abs, rms, frac_peak, frac_l2


def make_grid(npix, grid, margin):
    x = np.linspace(margin, npix - margin, grid)
    y = np.linspace(margin, npix - margin, grid)
    return x, y


def run_scan(model, scas, grid, stamp_size, scale, margin, aberrations):
    nparam = model.param_len
    if aberrations.shape[0] != nparam:
        raise ValueError(f"Expected {nparam} aberration values, got {aberrations.shape[0]}")

    xg, yg = make_grid(model.sca_size, grid, margin)
    rows = []
    maps = {}

    for sca in scas:
        max_abs_map = np.zeros((grid, grid), dtype=float)
        rms_map = np.zeros((grid, grid), dtype=float)
        frac_peak_map = np.zeros((grid, grid), dtype=float)
        frac_l2_map = np.zeros((grid, grid), dtype=float)

        for iy, y in enumerate(yg):
            for ix, x in enumerate(xg):
                star = piff.Star.makeTarget(
                    x=float(x),
                    y=float(y),
                    stamp_size=stamp_size,
                    scale=scale,
                    properties={"sca": int(sca)},
                ).withFlux(1.0, (0.0, 0.0))
                star = model.initialize(star)

                model_star = model.draw(
                    piff.Star(
                        star.data,
                        star.fit.newParams(
                            aberrations, params_var=np.zeros_like(aberrations), num=model._num
                        ),
                    )
                )

                prof = galsim.roman.getPSF(
                    int(sca),
                    model.filter,
                    SCA_pos=star.image_pos,
                    pupil_bin=piff.roman_psf.pupil_bin,
                    wcs=star.image.wcs,
                    extra_aberrations=model._make_extra_aberrations(aberrations),
                    wavelength=None if model.chromatic else model.bandpass.effective_wavelength,
                )
                direct_image = star.image.copy()
                prof.drawImage(
                    direct_image,
                    method=model._method,
                    center=star.image_pos,
                    bandpass=model.bandpass if model.chromatic else None,
                )

                max_abs, rms, frac_peak, frac_l2 = point_metrics(
                    model_star.image.array, direct_image.array
                )
                max_abs_map[iy, ix] = max_abs
                rms_map[iy, ix] = rms
                frac_peak_map[iy, ix] = frac_peak
                frac_l2_map[iy, ix] = frac_l2

        maps[sca] = {
            "x_grid": xg.copy(),
            "y_grid": yg.copy(),
            "max_abs": max_abs_map,
            "rms": rms_map,
            "frac_peak": frac_peak_map,
            "frac_l2": frac_l2_map,
        }

        rows.append(
            (
                sca,
                float(np.nanmedian(max_abs_map)),
                float(np.nanpercentile(max_abs_map, 95)),
                float(np.nanmax(max_abs_map)),
                float(np.nanmedian(frac_peak_map)),
                float(np.nanpercentile(frac_peak_map, 95)),
                float(np.nanmax(frac_peak_map)),
                float(np.nanmedian(frac_l2_map)),
                float(np.nanpercentile(frac_l2_map, 95)),
                float(np.nanmax(frac_l2_map)),
            )
        )
    return rows, maps


def print_summary(rows):
    print("\nPer-SCA bilinear-vs-direct mismatch summary:")
    print(
        "  SCA   med|max|   p95|max|   max|max|   med(max/peak)   p95(max/peak)"
        "   max(max/peak)   med(l2frac)   p95(l2frac)   max(l2frac)"
    )
    for row in sorted(rows, key=lambda r: r[0]):
        print(
            f"  {row[0]:3d}  {row[1]:10.4e} {row[2]:10.4e} {row[3]:10.4e}"
            f"  {row[4]:13.4e} {row[5]:13.4e} {row[6]:13.4e}"
            f"  {row[7]:11.4e} {row[8]:11.4e} {row[9]:11.4e}"
        )

    worst = max(rows, key=lambda r: r[6])
    print(
        "\nWorst SCA by peak-fraction max mismatch: "
        f"SCA {worst[0]} with max(max/peak)={worst[6]:.4e}"
    )


def save_maps_npz(path, maps):
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    payload = {}
    for sca, d in maps.items():
        payload[f"sca{sca}_x"] = d["x_grid"]
        payload[f"sca{sca}_y"] = d["y_grid"]
        payload[f"sca{sca}_max_abs"] = d["max_abs"]
        payload[f"sca{sca}_rms"] = d["rms"]
        payload[f"sca{sca}_frac_peak"] = d["frac_peak"]
        payload[f"sca{sca}_frac_l2"] = d["frac_l2"]
    np.savez(path, **payload)
    print(f"\nWrote mismatch maps to {path}")


def maybe_plot(path, maps, metric):
    if path is None:
        return
    import matplotlib.pyplot as plt
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if metric not in ("max_abs", "rms", "frac_peak", "frac_l2"):
        raise ValueError(f"Invalid metric {metric}")

    scas = sorted(maps)
    ncol = 6
    nrow = int(np.ceil(len(scas) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), squeeze=False, constrained_layout=True
    )

    vmin = min(np.nanmin(maps[sca][metric]) for sca in scas)
    vmax = max(np.nanmax(maps[sca][metric]) for sca in scas)

    for ax in axes.ravel():
        ax.set_visible(False)

    im = None
    for i, sca in enumerate(scas):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        data = maps[sca][metric]
        x = maps[sca]["x_grid"]
        y = maps[sca]["y_grid"]
        im = ax.imshow(
            data,
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"SCA {sca}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
        cbar.set_label(metric)
    fig.suptitle(f"Roman bilinear-vs-direct mismatch map: {metric}")
    fig.savefig(path, dpi=150)
    print(f"Wrote plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scas",
        default=None,
        help="SCA selection, e.g. '1-18' or '5,8,10'. Default: all 1..18.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=5,
        help="Grid points per axis per SCA (default: 5).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=64.0,
        help="Grid margin in pixels from each SCA edge (default: 64).",
    )
    parser.add_argument(
        "--stamp-size",
        type=int,
        default=25,
        help="Stamp size in pixels (default: 25).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.11,
        help="Pixel scale in arcsec/pixel (default: 0.11).",
    )
    parser.add_argument(
        "--pupil-bin",
        type=int,
        default=8,
        help="Roman pupil_bin for GalSim getPSF draws (default: 8).",
    )
    parser.add_argument(
        "--aberrations",
        default=None,
        help="Comma-separated z4..zN values (length max_zernike-3). Default all zeros.",
    )
    parser.add_argument(
        "--max-zernike",
        type=int,
        default=6,
        help="Roman max_zernike for diagnostic model (default: 6).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional output image path for 18-SCA metric map.",
    )
    parser.add_argument(
        "--plot-metric",
        default="frac_peak",
        choices=["max_abs", "rms", "frac_peak", "frac_l2"],
        help="Metric to visualize when --plot is set (default: frac_peak).",
    )
    parser.add_argument(
        "--save-npz",
        default=None,
        help="Optional .npz path to save all per-SCA mismatch maps.",
    )
    args = parser.parse_args()

    piff.roman_psf.pupil_bin = int(args.pupil_bin)
    model = piff.Roman(
        filter="H158",
        chromatic=False,
        max_zernike=args.max_zernike,
        aberration_interp="constant",
        aberration_prior_sigma=1.0e6,
    )

    scas = parse_scas(args.scas)
    aberrations = parse_aberrations(args.aberrations, model.param_len)
    print(
        "Running bilinear-vs-direct scan with "
        f"scas={scas}, grid={args.grid}, pupil_bin={args.pupil_bin}, "
        f"max_zernike={args.max_zernike}, aberrations={aberrations}"
    )

    rows, maps = run_scan(
        model=model,
        scas=scas,
        grid=args.grid,
        stamp_size=args.stamp_size,
        scale=args.scale,
        margin=args.margin,
        aberrations=aberrations,
    )
    print_summary(rows)
    if args.save_npz:
        save_maps_npz(args.save_npz, maps)
    maybe_plot(args.plot, maps, args.plot_metric)


if __name__ == "__main__":
    main()
