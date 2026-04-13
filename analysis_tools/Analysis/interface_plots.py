# -*- coding: utf-8 -*-
"""
plot_interface.py
-----------------
Parse tab-separated interface conductivity/Ea data and produce
4 publication-quality figures (one per spatial segment).
Each figure: left y = sigma (log, blue), right y = Ea (linear, red).

Input format (tab-separated):
  - Condition label line  (e.g. "300K", "900K250ps")
  - z header line         (z   6  20  20  44  44  58   6  58)
  - sigma row             (sigma (S/cm)  val  err  val  err ...)
  - Ea row                (Ea (eV)       val  err  val  err ...)
  - blank line between blocks

Usage:
  python plot_interface.py                       # uses example_input.txt
  python plot_interface.py mydata.txt            # custom file
  python plot_interface.py mydata.txt out_dir/   # custom file + output dir
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- fixed aesthetics ---------------------------------------------------------
COLOR_SIGMA  = "#2166AC"   # blue  - sigma (left axis)
COLOR_EA     = "#C0392B"   # red   - Ea    (right axis)
MARKER_SIGMA = "o"         # circle
MARKER_EA    = "^"         # triangle
COLOR_X      = "black"

SEGMENT_LABELS = ["z = 6-20 A", "z = 20-44 A", "z = 44-58 A", "z = 6-58 A (full)"]
SEG_NAMES      = ["6-20A", "20-44A", "44-58A", "6-58A_full"]

FIG_WIDTH    = 6.0
FIG_HEIGHT   = 4.2
FONT_FAMILY  = "Arial"
FONTSIZE     = 9
CAPSIZE      = 3
ELINEWIDTH   = 1.2
LINEWIDTH    = 1.2
MARKERSIZE   = 6
MARKEREDGE   = 1.4
DPI          = 300
# ------------------------------------------------------------------------------


def parse_file(path):
    """Return list of dicts: {label, sigma[4], sigma_err[4], ea[4], ea_err[4]}"""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    raw = re.sub(r'\n[ \t]+\n', '\n\n', raw)
    blocks = [b.strip() for b in re.split(r'\n{2,}', raw) if b.strip()]
    records = []

    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        label = lines[0].strip().split('\t')[0].strip()

        sigma_vals = ea_vals = None
        for line in lines[1:]:
            cols      = line.split('\t')
            key       = cols[0].strip().lower()
            key_ascii = key.encode("ascii", errors="ignore").decode()
            nums_str  = [c.strip() for c in cols[1:] if c.strip()]
            try:
                nums = [float(x) for x in nums_str]
            except ValueError:
                continue

            if 's/cm' in key_ascii or key_ascii.startswith('s('):
                sigma_vals = nums
            elif key_ascii.startswith('ea') or 'ev' in key_ascii:
                ea_vals = nums

        if sigma_vals is None or ea_vals is None:
            print("  Warning: could not parse block '{}' - skipping.".format(label))
            continue
        if len(sigma_vals) < 8 or len(ea_vals) < 8:
            print("  Warning: block '{}' has fewer than 8 columns - skipping.".format(label))
            continue

        def pair(vals, i):
            return vals[2 * i], vals[2 * i + 1]

        sigma   = [pair(sigma_vals, i)[0] for i in range(4)]
        sigma_e = [pair(sigma_vals, i)[1] for i in range(4)]
        ea      = [pair(ea_vals,    i)[0] for i in range(4)]
        ea_e    = [pair(ea_vals,    i)[1] for i in range(4)]

        records.append(dict(label=label,
                            sigma=sigma,   sigma_err=sigma_e,
                            ea=ea,         ea_err=ea_e))
    return records


def setup_rcparams():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size":          FONTSIZE,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.minor.width":  0.5,
        "ytick.minor.width":  0.5,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        False,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
    })


def apply_log_ticks(ax, color, n_ticks=5):
    ymin, ymax = ax.get_ylim()
    if ymin <= 0: return

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='all', numticks=n_ticks))

    formatter = ticker.LogFormatterSciNotation(labelOnlyBase=False)
    ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.tick_params(axis='y', which='major', labelsize=FONTSIZE, labelcolor=color, colors=color, direction='in')
    
    ax.set_ylim(ymin * 0.9, ymax * 1.1)


def make_segment_figure(records, si, out_path):
    """One figure per segment."""
    setup_rcparams()
    n_cond  = len(records)
    x       = np.arange(n_cond)
    xlabels = [r["label"] for r in records]

    fig, ax_s = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax_e = ax_s.twinx()

    # ---- left axis: sigma (blue, solid, circle) -----------------------------
    sig_vals = np.array([r["sigma"][si]     for r in records])
    sig_errs = np.array([r["sigma_err"][si] for r in records])

    ax_s.errorbar(x, sig_vals,
                  yerr=sig_errs,
                  fmt=MARKER_SIGMA,
                  color=COLOR_SIGMA,
                  markerfacecolor="white",
                  markeredgewidth=MARKEREDGE,
                  markersize=MARKERSIZE,
                  linewidth=LINEWIDTH,
                  linestyle="-",
                  capsize=CAPSIZE,
                  elinewidth=ELINEWIDTH,
                  capthick=ELINEWIDTH,
                  label=r"$\sigma$",
                  zorder=3)
    ax_s.fill_between(x, sig_vals - sig_errs, sig_vals + sig_errs,
                      color=COLOR_SIGMA, alpha=0.10, zorder=2)

    ax_s.set_yscale("log")
    apply_log_ticks(ax_s, COLOR_SIGMA)

    ax_s.set_ylabel(r"$\sigma$ (S cm$^{-1}$)",
                    fontsize=FONTSIZE, color=COLOR_SIGMA)
    ax_s.spines['left'].set_color(COLOR_SIGMA)
    ax_s.spines['left'].set_linewidth(0.8)

    # ---- right axis: Ea (red, solid, triangle) -------------------------------
    ea_vals = np.array([r["ea"][si]     for r in records])
    ea_errs = np.array([r["ea_err"][si] for r in records])

    ax_e.errorbar(x, ea_vals,
                  yerr=ea_errs,
                  fmt=MARKER_EA,
                  color=COLOR_EA,
                  markerfacecolor="white",
                  markeredgewidth=MARKEREDGE,
                  markersize=MARKERSIZE,
                  linewidth=LINEWIDTH,
                  linestyle="-",       # solid, no dash
                  capsize=CAPSIZE,
                  elinewidth=ELINEWIDTH,
                  capthick=ELINEWIDTH,
                  label=r"$E_a$",
                  zorder=3)
    ax_e.fill_between(x, ea_vals - ea_errs, ea_vals + ea_errs,
                      color=COLOR_EA, alpha=0.10, zorder=2)

    ax_e.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=4))
    ax_e.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_e.set_ylabel(r"$E_a$ (eV)", fontsize=FONTSIZE, color=COLOR_EA)
    ax_e.tick_params(axis='y',
                     labelsize=FONTSIZE, labelcolor=COLOR_EA,
                     which='both', colors=COLOR_EA)
    ax_e.spines['right'].set_color(COLOR_EA)
    ax_e.spines['right'].set_linewidth(0.8)

    # ---- x axis (black) ------------------------------------------------------
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(xlabels, rotation=35, ha="right",
                         fontsize=FONTSIZE, color=COLOR_X)
    ax_s.tick_params(axis='x', labelsize=FONTSIZE, colors=COLOR_X)
    ax_s.spines['bottom'].set_color(COLOR_X)
    ax_s.spines['top'].set_color(COLOR_X)
    ax_s.set_xlim(-0.6, n_cond - 0.4)
    ax_s.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)

    # ---- title + legend (top center, outside plot area) ----------------------
    ax_s.set_title(SEGMENT_LABELS[si], fontsize=FONTSIZE + 1, pad=5, color=COLOR_X)

    lines_s, labs_s = ax_s.get_legend_handles_labels()
    lines_e, labs_e = ax_e.get_legend_handles_labels()
    ax_s.legend(lines_s + lines_e, labs_s + labs_e,
                fontsize=FONTSIZE, framealpha=0.8,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2,
                handlelength=1.5,
                columnspacing=1.0,
                borderpad=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> {}".format(out_path))


def main():
    txt_file = sys.argv[1] if len(sys.argv) > 1 else "interface_plots.txt"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    print("Reading: {}".format(txt_file))
    records = parse_file(txt_file)
    if not records:
        print("No valid data blocks found. Check file format.")
        return
    print("  Parsed {} conditions: {}".format(
        len(records), [r["label"] for r in records]))

    for si, name in enumerate(SEG_NAMES):
        out_path = os.path.join(out_dir, "fig_seg_{}.png".format(name))
        make_segment_figure(records, si, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
