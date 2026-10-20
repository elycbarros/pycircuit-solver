"""
plots.py — Matplotlib-based visualizations for pycircuit-solver results.

Functions
---------
plot_power_transfer(result)
    Plot the P_L vs R_L power-transfer curve from a :class:`TheveninResult`.

plot_monte_carlo(result, node)
    Plot the voltage distribution histogram for a specific node from a
    :class:`MonteCarloResult`.

plot_monte_carlo_all(result)
    Side-by-side histograms for all non-ground nodes.

All functions return the ``matplotlib.figure.Figure`` object so callers can
further customise or save it without showing it interactively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import sys
import numpy as np
import matplotlib

# In headless/CI environments without display, avoid GUI backend issues
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if TYPE_CHECKING:
    from circuit_solver.engine.thevenin import TheveninResult
    from circuit_solver.engine.monte_carlo import MonteCarloResult

__all__ = [
    "plot_power_transfer",
    "plot_monte_carlo",
    "plot_monte_carlo_all",
    "plot_bode",
]

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "axes.facecolor": "#0f1117",
    "figure.facecolor": "#0f1117",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "text.color": "#e0e0e0",
    "grid.color": "#333",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
}

_ACCENT = "#00c8ff"
_HIGHLIGHT = "#ff6b6b"
_GREEN = "#4ade80"


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(_STYLE["axes.facecolor"])
    ax.grid(True, color=_STYLE["grid.color"], linestyle="--", linewidth=0.5)
    ax.tick_params(colors=_STYLE["xtick.color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_STYLE["axes.edgecolor"])


# ---------------------------------------------------------------------------
# Power transfer curve
# ---------------------------------------------------------------------------

def plot_power_transfer(
    result: "TheveninResult",
    *,
    title: str = "Maximum Power Transfer",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot load power P_L vs load resistance R_L.

    A vertical dashed line marks ``R_L = R_th`` (maximum power point).

    Parameters
    ----------
    result : TheveninResult
    title : str
    show : bool
        If True, call ``plt.show()`` after creating the figure.
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(_STYLE["figure.facecolor"])
        _apply_style(ax)

        rl = result.R_L_sweep
        pl = result.P_L_sweep

        ax.semilogx(rl, pl * 1e3, color=_ACCENT, linewidth=2.5, label="$P_L$ (mW)")

        # Mark maximum power point
        ax.axvline(result.R_th, color=_HIGHLIGHT, linestyle="--", linewidth=1.5,
                   label=f"$R_L = R_{{th}} = {result.R_th:.2f}$ Ω")
        ax.axhline(result.P_max * 1e3, color=_GREEN, linestyle=":", linewidth=1.2,
                   label=f"$P_{{max}} = {result.P_max*1e3:.4f}$ mW")

        # Annotate peak
        ax.scatter([result.R_th], [result.P_max * 1e3],
                   color=_HIGHLIGHT, s=80, zorder=5)
        ax.annotate(
            f" $P_{{max}}$\n {result.P_max*1e3:.4f} mW",
            xy=(result.R_th, result.P_max * 1e3),
            xytext=(result.R_th * 2.5, result.P_max * 0.9 * 1e3),
            color=_HIGHLIGHT,
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=_HIGHLIGHT, lw=1.2),
        )

        ax.set_xlabel("Load Resistance $R_L$ (Ω)", fontsize=11, color="#e0e0e0")
        ax.set_ylabel("Load Power $P_L$ (mW)", fontsize=11, color="#e0e0e0")
        ax.set_title(title, fontsize=13, color="#ffffff", fontweight="bold", pad=12)
        ax.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        info = (
            f"$V_{{th}}$ = {result.V_th:.4f} V   |   "
            f"$R_{{th}}$ = {result.R_th:.4f} Ω   |   "
            f"$I_N$ = {result.I_N:.6f} A"
        )
        fig.text(0.5, 0.01, info, ha="center", va="bottom",
                 fontsize=8, color="#888")

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo histograms
# ---------------------------------------------------------------------------

def plot_monte_carlo(
    result: "MonteCarloResult",
    node: str,
    *,
    bins: int = 50,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot voltage distribution histogram for a single node.

    Parameters
    ----------
    result : MonteCarloResult
    node : str
        Node label to plot.
    bins : int
    title : str | None
    show : bool
    save_path : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if node not in result.node_labels:
        raise KeyError(f"Node '{node}' not in result. Available: {result.node_labels}")

    idx = result.node_labels.index(node)
    data = result.samples[:, idx]
    mu = result.mean[idx]
    sigma = result.std[idx]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(_STYLE["figure.facecolor"])
        _apply_style(ax)

        n_bins, edges, patches = ax.hist(
            data, bins=bins, color=_ACCENT, alpha=0.75, edgecolor="#0f1117", linewidth=0.3
        )
        # Shade ±1σ region
        lo, hi = mu - sigma, mu + sigma
        for patch, left, right in zip(patches, edges[:-1], edges[1:]):
            if lo <= left and right <= hi:
                patch.set_facecolor(_GREEN)
                patch.set_alpha(0.85)

        ax.axvline(mu, color=_HIGHLIGHT, linewidth=2.0, label=f"μ = {mu:.5f} V")
        ax.axvline(result.p5[idx], color="#f59e0b", linewidth=1.2, linestyle="--",
                   label=f"P5  = {result.p5[idx]:.5f} V")
        ax.axvline(result.p95[idx], color="#f59e0b", linewidth=1.2, linestyle="--",
                   label=f"P95 = {result.p95[idx]:.5f} V")

        ax.set_xlabel(f"$V({node})$ (V)", fontsize=11, color="#e0e0e0")
        ax.set_ylabel("Count", fontsize=11, color="#e0e0e0")
        _title = title or f"Monte Carlo — $V({node})$ Distribution"
        ax.set_title(_title, fontsize=13, color="#ffffff", fontweight="bold", pad=12)
        ax.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        stats = (
            f"n={result.n_samples}  failed={result.n_failed}  "
            f"σ={sigma:.5f} V  min={result.min[idx]:.5f} V  max={result.max[idx]:.5f} V"
        )
        fig.text(0.5, 0.01, stats, ha="center", va="bottom", fontsize=8, color="#888")
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    return fig


def plot_monte_carlo_all(
    result: "MonteCarloResult",
    *,
    bins: int = 40,
    title: str = "Monte Carlo — All Nodes",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot voltage distributions for all non-ground nodes in a grid.

    Parameters
    ----------
    result : MonteCarloResult
    bins : int
    title : str
    show : bool
    save_path : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(result.node_labels)
    if n == 0:
        raise ValueError("No nodes to plot.")

    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        fig.patch.set_facecolor(_STYLE["figure.facecolor"])
        fig.suptitle(title, fontsize=14, color="#ffffff", fontweight="bold", y=1.01)

        # Flatten axes for uniform iteration
        ax_list = np.array(axes).flatten() if n > 1 else [axes]

        for i, (node, ax) in enumerate(zip(result.node_labels, ax_list)):
            _apply_style(ax)
            data = result.samples[:, i]
            mu = result.mean[i]
            sigma = result.std[i]

            ax.hist(data, bins=bins, color=_ACCENT, alpha=0.75,
                    edgecolor="#0f1117", linewidth=0.3)
            ax.axvline(mu, color=_HIGHLIGHT, linewidth=1.8,
                       label=f"μ={mu:.4f} V\nσ={sigma:.4f} V")
            ax.set_title(f"$V({node})$", fontsize=11, color="#ccc")
            ax.set_xlabel("Voltage (V)", fontsize=9, color="#aaa")
            ax.set_ylabel("Count", fontsize=9, color="#aaa")
            ax.legend(framealpha=0.1, fontsize=8)

        # Hide unused subplots
        for ax in ax_list[n:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    return fig


# ---------------------------------------------------------------------------
# Bode plot (Frequency Response)
# ---------------------------------------------------------------------------

def plot_bode(
    frequencies: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
    *,
    title: str = "Bode Diagram (Frequency Response)",
    target_label: str = "Output",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Bode magnitude and phase diagrams.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in Hz.
    mag_db : np.ndarray
        Magnitude in dB.
    phase_deg : np.ndarray
        Phase in degrees.
    title : str
    target_label : str
    show : bool
    save_path : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    with plt.rc_context(_STYLE):
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.patch.set_facecolor(_STYLE["figure.facecolor"])
        _apply_style(ax_mag)
        _apply_style(ax_phase)

        fig.suptitle(title, fontsize=13, color="#ffffff", fontweight="bold")

        # Magnitude plot
        ax_mag.semilogx(frequencies, mag_db, color=_ACCENT, linewidth=2.0, label=f"|{target_label}| (dB)")
        ax_mag.set_ylabel("Magnitude (dB)", fontsize=10, color="#e0e0e0")
        ax_mag.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        # Phase plot
        ax_phase.semilogx(frequencies, phase_deg, color="#f59e0b", linewidth=2.0, label=f"∠ {target_label} (°)")
        ax_phase.set_xlabel("Frequency (Hz)", fontsize=10, color="#e0e0e0")
        ax_phase.set_ylabel("Phase (°)", fontsize=10, color="#e0e0e0")
        ax_phase.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    return fig


def plot_transient(
    time: np.ndarray,
    node_voltages: dict[str, np.ndarray],
    nodes_to_plot: Sequence[str] | None = None,
    branch_currents: dict[str, np.ndarray] | None = None,
    elements_to_plot: Sequence[str] | None = None,
    *,
    title: str = "Transient Waveforms",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot time-domain voltage and current waveforms.

    Parameters
    ----------
    time : np.ndarray
        Time points in seconds.
    node_voltages : dict[str, np.ndarray]
        Mapping of node name to voltage waveform array.
    nodes_to_plot : Sequence[str] | None, optional
        Specific node labels to display. If None, plots all nodes.
    branch_currents : dict[str, np.ndarray] | None, optional
        Mapping of element name to branch current waveform array.
    elements_to_plot : Sequence[str] | None, optional
        Specific elements whose branch currents to plot.
    title : str, optional
        Figure title.
    show : bool, optional
        Display interactive window if True.
    save_path : str | None, optional
        Save plot image to file if specified.

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_currents = bool(branch_currents and elements_to_plot)
    n_plots = 2 if has_currents else 1

    colors = ["#38bdf8", "#34d399", "#f59e0b", "#a78bfa", "#f43f5e", "#10b981", "#60a5fa"]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(n_plots, 1, figsize=(9, 4.5 * n_plots), sharex=True)
        if n_plots == 1:
            ax_v = axes
            ax_i = None
        else:
            ax_v, ax_i = axes

        fig.patch.set_facecolor(_STYLE["figure.facecolor"])
        _apply_style(ax_v)
        fig.suptitle(title, fontsize=13, color="#ffffff", fontweight="bold")

        # Voltage plot
        target_nodes = nodes_to_plot or list(node_voltages.keys())
        for idx, node in enumerate(target_nodes):
            if node in node_voltages:
                c = colors[idx % len(colors)]
                ax_v.plot(time, node_voltages[node], color=c, linewidth=2.0, label=f"V({node})")

        ax_v.set_ylabel("Voltage (V)", fontsize=10, color="#e0e0e0")
        ax_v.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        # Current plot if requested
        if ax_i is not None and branch_currents and elements_to_plot:
            _apply_style(ax_i)
            for idx, el_name in enumerate(elements_to_plot):
                if el_name in branch_currents:
                    c = colors[(idx + 3) % len(colors)]
                    ax_i.plot(time, branch_currents[el_name], color=c, linewidth=2.0, label=f"I({el_name})")

            ax_i.set_ylabel("Current (A)", fontsize=10, color="#e0e0e0")
            ax_i.legend(framealpha=0.15, edgecolor="#555", fontsize=9)

        # Format X axis
        target_ax = ax_i if ax_i is not None else ax_v
        target_ax.set_xlabel("Time (s)", fontsize=10, color="#e0e0e0")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    return fig

