"""
Cyberpunk EnKF Results Dashboard  —  port 8060
Dynamic viewport system: each panel is independent; add/remove without reloading others.
"""
import pathlib
import pickle
import threading
import uuid

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, callback, Input, Output, State, MATCH, ALL, Patch, ctx, no_update

# ─────────────────────────────────────────────────────────────────
# 0. Discovery & constants
# ─────────────────────────────────────────────────────────────────
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def _discover_pkls() -> list:
    return sorted(RESULTS_DIR.glob("l63_benchmark_results_*.pkl"))


PKL_FILES = _discover_pkls()
PKL_NAMES = [p.name for p in PKL_FILES]

MODELS = ["Lorenz63", "DenseNN", "ResDenseNN", "LSTMNN", "RNN_relu", "RNN_tanh"]
VAR_NAMES = ["x", "y", "z"]

NEON = {
    "Lorenz63":   "#00ff9f",
    "DenseNN":    "#ff00ff",
    "ResDenseNN": "#00d4ff",
    "LSTMNN":     "#ff6600",
    "RNN_relu":   "#ffcc00",
    "RNN_tanh":   "#ff0055",
}
REF =  ["#00f0ff", "#ff2bd6", "#39ffb6", "#ffcc00"]

NEON = {
    "Lorenz63":   "#b9c7bd",
    "DenseNN":    "#ff6600",
    "ResDenseNN": "#ffcc00",
    "LSTMNN":     '#7F77DD',
    "RNN_relu":   "#1D9E75",
    "RNN_tanh":   "#39ffb6"
}
SWEEP_VALS = {
    "b2": [5, 10, 20, 40, 50, 100],
    "b3": [5, 10, 20, 50, 100],
    "b4": [1.0, 0.67, 0.33],
    "b5": [0.1, 0.5, 1.0, 2.0, 3.0],
}
SWEEP_LABEL = {
    "b2": "M (forecast steps)",
    "b3": "Ne (ensemble size)",
    "b4": "p (obs fraction)",
    "b5": "σ_obs (obs noise)",
}
DROP_T = 50  # spin-up cycles to drop in aggregation

PLOT_TYPES = [
    {"label": "RMSE Comparison",      "value": "rmse_comparison"},
    {"label": "Ensemble Spread",      "value": "spread_comparison"},
    {"label": "Spread vs RMSE",       "value": "spread_vs_rmse"},
    {"label": "Rank Histograms",      "value": "talagrand_grid"},
    {"label": "Spaghetti Ensemble",   "value": "spaghetti_multi"},
    {"label": "Spread Reduction",     "value": "spread_reduction_multi"},
    {"label": "Trajectory",           "value": "trajectory_comparison"},
    {"label": "Sweep: RMSE",          "value": "sweep_rmse"},
    {"label": "Agg: Boxplot",         "value": "agg_boxplot"},
    {"label": "Agg: Violin",          "value": "agg_violin"},
    {"label": "Agg: Mean ± Std",      "value": "agg_mean_std"},
    {"label": "Agg: Spaghetti+Mean",  "value": "agg_sweep_line"},
]

# ─────────────────────────────────────────────────────────────────
# 1. Cache layer
# ─────────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_file_cache: dict = {}
_file_cache_order: list = []
MAX_CACHED_FILES = 2

_agg_cache: dict = {"ready": False, "files_loaded": 0, "total": 0}


def get_pkl_data(filename: str) -> dict:
    with _cache_lock:
        if filename in _file_cache:
            _file_cache_order.remove(filename)
            _file_cache_order.append(filename)
            return _file_cache[filename]
        if len(_file_cache) >= MAX_CACHED_FILES:
            evict = _file_cache_order.pop(0)
            del _file_cache[evict]
    # Load outside lock to avoid blocking callbacks during slow disk I/O
    with open(RESULTS_DIR / filename, "rb") as f:
        data = pickle.load(f)
    with _cache_lock:
        _file_cache[filename] = data
        _file_cache_order.append(filename)
    return data


def _load_agg_cache_background():
    pkls = _discover_pkls()
    total = len(pkls)
    with _cache_lock:
        _agg_cache["total"] = total
        _agg_cache["b1"] = {m: {"errorf_seeds": [], "errora_seeds": [], "spread_seeds": []}
                             for m in MODELS}
        for bench, vals in SWEEP_VALS.items():
            _agg_cache[bench] = {
                v: {m: {"errorf_seeds": [], "errora_seeds": []} for m in MODELS}
                for v in vals
            }

    for pkl_path in pkls:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            # b1 baseline
            for model, res in data.get("b1", {}).items():
                if model not in MODELS:
                    continue
                ef_arr = res.get("errorf", [])
                ea_arr = res.get("errora", [])
                sp_arr = res.get("spread", [])
                ef = float(np.nanmean(ef_arr[DROP_T:])) if len(ef_arr) > DROP_T else float("nan")
                ea = float(np.nanmean(ea_arr[DROP_T:])) if len(ea_arr) > DROP_T else float("nan")
                es = float(np.nanmean(sp_arr[DROP_T:])) if len(sp_arr) > DROP_T else float("nan")
                with _cache_lock:
                    _agg_cache["b1"][model]["errorf_seeds"].append(ef)
                    _agg_cache["b1"][model]["errora_seeds"].append(ea)
                    _agg_cache["b1"][model]["spread_seeds"].append(es)

            # b2–b5 sweeps
            for bench, vals in SWEEP_VALS.items():
                res_dict = data.get(bench, {}).get("results", {})
                for v in vals:
                    pv = res_dict.get(v, {})
                    for model in MODELS:
                        r = pv.get(model, {})
                        ef_arr = r.get("errorf", []) if r else []
                        ea_arr = r.get("errora", []) if r else []
                        ef = float(np.nanmean(ef_arr[DROP_T:])) if len(ef_arr) > DROP_T else float("nan")
                        ea = float(np.nanmean(ea_arr[DROP_T:])) if len(ea_arr) > DROP_T else float("nan")
                        with _cache_lock:
                            _agg_cache[bench][v][model]["errorf_seeds"].append(ef)
                            _agg_cache[bench][v][model]["errora_seeds"].append(ea)
        except Exception:
            pass

        with _cache_lock:
            _agg_cache["files_loaded"] += 1

    with _cache_lock:
        _agg_cache["ready"] = True


threading.Thread(target=_load_agg_cache_background, daemon=True).start()

# ─────────────────────────────────────────────────────────────────
# 2. Cyberpunk theme
# ─────────────────────────────────────────────────────────────────
_BG      = "#0a0e14" #"#0a0a0f"
_PLOT_BG = "#0d1117" #"#0d0d1a"
_GRID    =  "rgba(0,240,255,0.12)" # "#1a1a2e"
_LINE    =  "rgba(255,43,214,0.35)" #"#2a2a4e"
_TEXT    = "#c9d1d9" #"#c0c0d0"
_GREEN   = "#00ff9f"
_BLUE    = "#00f0ff"
_GREEN   = _BLUE

pio.templates["cyberpunk"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor=_BG,
    plot_bgcolor=_PLOT_BG,
    #font=dict(color=_TEXT, family="Courier New, monospace", size=11),
    font=dict(color=_TEXT, family="JetBrains Mono, ui-monospace, monospace", size=11),
    xaxis=dict(gridcolor=_GRID, linecolor=_LINE, showgrid=True,
               zeroline=False, zerolinecolor=_LINE),
    yaxis=dict(gridcolor=_GRID, linecolor=_LINE, showgrid=True,
               zeroline=False, zerolinecolor=_LINE),
    legend=dict(bgcolor="rgba(10,10,20,0.85)", bordercolor=_LINE, borderwidth=1,
                font=dict(color=_TEXT, size=10)),
    margin=dict(l=50, r=20, t=50, b=40),
    title=dict(font=dict(color=_GREEN, family="Courier New, monospace", size=13)),
    colorway=list(NEON.values()),
))

# Inline style dicts
S_PAGE    = {"display": "flex", "flexDirection": "row", "backgroundColor": _BG,
             "minHeight": "100vh", "color": _TEXT, "fontFamily": "Courier New, monospace"}
S_SIDEBAR = {"width": "260px", "minWidth": "260px", "backgroundColor": "#0d0d1a",
             "borderRight": f"1px solid {_GREEN}44", "padding": "16px",
             "overflowY": "auto", "height": "100vh", "position": "sticky",
             "top": 0, "boxSizing": "border-box"}
S_GRID    = {"flex": "1", "display": "flex", "flexWrap": "wrap", "gap": "10px",
             "padding": "10px", "alignContent": "flex-start", "minWidth": 0}
S_CARD    = {"backgroundColor": "#0d0d1a", "border": f"1px solid {_LINE}",
             "borderRadius": "4px", "boxShadow": f"0 0 12px {_GREEN}1a",
             "padding": "8px", "boxSizing": "border-box",
             "width": "calc(50% - 5px)", "minWidth": "500px"}
S_ROW     = {"display": "flex", "alignItems": "flex-end", "gap": "6px",
             "marginBottom": "6px", "flexWrap": "wrap"}
S_LABEL   = {"fontSize": "9px", "color": f"{_GREEN}99", "marginBottom": "2px",
             "letterSpacing": "1px"}
S_BTN     = {"backgroundColor": "transparent", "color": _GREEN,
             "border": f"1px solid {_GREEN}", "cursor": "pointer",
             "fontFamily": "Courier New, monospace", "padding": "5px 14px",
             "borderRadius": "3px", "fontSize": "12px"}
S_CLOSE   = {**S_BTN, "color": "#ff0055", "border": "1px solid #ff0055",
             "marginLeft": "auto", "padding": "3px 9px", "fontSize": "13px",
             "alignSelf": "flex-end"}
S_HR      = {"borderColor": f"{_GREEN}33", "margin": "8px 0"}

CYBERPUNK_CSS = f"""
* {{ box-sizing: border-box; }}
::-webkit-scrollbar {{ width: 4px; background: {_BG}; }}
::-webkit-scrollbar-thumb {{ background: {_GREEN}44; border-radius: 2px; }}
.Select-control {{ background-color: #0a0a14 !important;
    border-color: {_LINE} !important; color: {_TEXT} !important; }}
.Select-value-label {{ color: {_TEXT} !important; }}
.Select-placeholder {{ color: #555 !important; }}
.Select-arrow {{ border-top-color: {_TEXT} !important; }}
.Select-menu-outer {{ background-color: #0d0d1a !important;
    border-color: {_LINE} !important; z-index: 9999 !important; }}
.VirtualizedSelectOption, .Select-option {{ color: {_TEXT} !important; background: #0d0d1a; }}
.VirtualizedSelectFocusedOption, .Select-option.is-focused {{ background: #1a1a3e !important; }}
.Select-option.is-selected {{ background: #0f2020 !important; color: {_GREEN} !important; }}
.rc-slider-rail {{ background: {_LINE}; }}
.rc-slider-track {{ background: {_GREEN}55; }}
.rc-slider-handle {{ border-color: {_GREEN} !important; background: {_BG} !important; }}
.rc-slider-handle:hover {{ box-shadow: 0 0 6px {_GREEN}; }}
input[type='checkbox'] {{ accent-color: {_GREEN}; }}
.js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
.js-plotly-plot .plotly .modebar-btn path {{ fill: {_TEXT} !important; opacity: 0.7; }}
"""

# ─────────────────────────────────────────────────────────────────
# 3. Pure numpy helpers
# ─────────────────────────────────────────────────────────────────

def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _make_empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="cyberpunk",
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(color=_GREEN, size=12))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _cfg_title(cfg) -> str:
    if cfg is None:
        return ""
    loc_s  = f"loc r={cfg.loc_radius}" if getattr(cfg, "use_localization", False) else "no loc"
    infl_s = f"λ={cfg.infl_factor}"    if getattr(cfg, "use_inflation",    False) else "no infl"
    return (f"Ne={getattr(cfg,'Ne','?')} | M={getattr(cfg,'M','?')} | "
            f"p={getattr(cfg,'p','?')} | σ={getattr(cfg,'sig_obs','?')} | {loc_s} | {infl_s}")


def _compute_rank_histogram(Xf_all, xt_traj):
    T, Nx, Ne = Xf_all.shape
    ranks = []
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            continue
        for v in range(Nx):
            ranks.append(int(np.searchsorted(np.sort(Xf_all[k, v, :]), xt_traj[k, v])))
    return np.array(ranks, dtype=int)


def _compute_rank_histogram_per_var(Xf_all, xt_traj):
    T, Nx, Ne = Xf_all.shape
    rpv = {v: [] for v in range(Nx)}
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            continue
        for v in range(Nx):
            rpv[v].append(int(np.searchsorted(np.sort(Xf_all[k, v, :]), xt_traj[k, v])))
    return {v: np.array(r, dtype=int) for v, r in rpv.items()}


# ─────────────────────────────────────────────────────────────────
# 4. Plot functions  (all return go.Figure)
# ─────────────────────────────────────────────────────────────────

def fig_rmse_comparison(results: dict, models: list, cfg=None) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Forecast RMSE", "Analysis RMSE"],
                        shared_yaxes=False)
    T = max((len(r.get("errorf", [])) for r in results.values()), default=1)
    cycles = list(range(T))
    for name in models:
        res = results.get(name)
        if res is None:
            continue
        color = NEON.get(name, "#888")
        dash  = "dash" if name == "Lorenz63" else "solid"
        width = 2.0   if name == "Lorenz63" else 1.4
        fig.add_trace(go.Scatter(x=cycles, y=res.get("errorf", []).tolist(),
                                  name=name, legendgroup=name,
                                  line=dict(color=color, dash=dash, width=width)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=cycles, y=res.get("errora", []).tolist(),
                                  name=name, legendgroup=name, showlegend=False,
                                  line=dict(color=color, dash=dash, width=width)),
                      row=1, col=2)
    fig.update_layout(template="cyberpunk",
                      title=f"RMSE Comparison — {_cfg_title(cfg)}" if cfg else "RMSE Comparison",
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(title_text="DA Cycle")
    fig.update_yaxes(title_text="RMSE", col=1)
    return fig


def fig_spread_comparison(results: dict, models: list, cfg=None) -> go.Figure:
    fig = go.Figure()
    T = max((len(r.get("spread", [])) for r in results.values()), default=1)
    cycles = list(range(T))
    for name in models:
        res = results.get(name)
        if res is None:
            continue
        color = NEON.get(name, "#888")
        fig.add_trace(go.Scatter(x=cycles, y=res.get("spread", []).tolist(), name=name,
                                  line=dict(color=color,
                                            dash="dash" if name == "Lorenz63" else "solid",
                                            width=1.4)))
    fig.update_layout(template="cyberpunk",
                      title=f"Ensemble Spread — {_cfg_title(cfg)}" if cfg else "Ensemble Spread",
                      xaxis_title="DA Cycle", yaxis_title="Mean Std",
                      legend=dict(orientation="h", y=-0.18))
    return fig


def fig_spread_vs_rmse(results: dict, models: list) -> go.Figure:
    fig = go.Figure()
    pts_x, pts_y = [], []
    for name in models:
        res = results.get(name)
        if res is None:
            continue
        ms = float(np.nanmean(res.get("spread", [np.nan])))
        mf = float(np.nanmean(res.get("errorf", [np.nan])))
        if not (np.isfinite(ms) and np.isfinite(mf)):
            continue
        color = NEON.get(name, "#888")
        fig.add_trace(go.Scatter(
            x=[ms], y=[mf], name=name, mode="markers+text",
            text=[name], textposition="top right",
            textfont=dict(color=color, size=9),
            marker=dict(color=color, size=11,
                        symbol="square" if name == "Lorenz63" else "circle",
                        line=dict(color="white", width=1)),
        ))
        pts_x.append(ms); pts_y.append(mf)
    if pts_x and pts_y:
        lo = min(min(pts_x), min(pts_y)) * 0.9
        hi = max(max(pts_x), max(pts_y)) * 1.1
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name="Ideal (spread=RMSE)",
                                  mode="lines",
                                  line=dict(color="white", dash="dash", width=1)))
    fig.update_layout(template="cyberpunk", title="Spread vs RMSE — calibration check",
                      xaxis_title="Mean Ensemble Spread", yaxis_title="Mean Forecast RMSE")
    return fig


def fig_talagrand_grid(results: dict, models: list, cfg=None,
                        ens_key: str = "Xf_all") -> go.Figure:
    active = [m for m in models if m in results and ens_key in results[m]]
    if not active:
        return _make_empty_fig("No ensemble data available for selected models")
    Ne  = results[active[0]][ens_key].shape[2]
    Nx  = 3
    n_r = len(active)
    fig = make_subplots(rows=n_r, cols=Nx + 1,
                        column_titles=VAR_NAMES + ["All vars"],
                        row_titles=active,
                        vertical_spacing=0.04, horizontal_spacing=0.04)
    bins = np.arange(Ne + 2) - 0.5
    uniform = 1.0 / (Ne + 1)
    for i, name in enumerate(active):
        res   = results[name]
        color = NEON.get(name, "#888")
        rgba  = _hex_rgba(color, 0.75)
        rpv   = _compute_rank_histogram_per_var(res[ens_key], res["xt_traj"])
        r_all = _compute_rank_histogram(res[ens_key], res["xt_traj"])
        for v in range(Nx):
            r = rpv[v]
            cnt, _ = np.histogram(r, bins=bins, density=True) if len(r) > 0 \
                     else (np.zeros(Ne + 1), bins)
            fig.add_trace(go.Bar(x=list(range(Ne + 1)), y=cnt.tolist(),
                                  name=name, legendgroup=name, showlegend=(i == 0 and v == 0),
                                  marker_color=rgba, marker_line_color="#000",
                                  marker_line_width=0.4),
                          row=i + 1, col=v + 1)
            fig.add_hline(y=uniform, line_dash="dash", line_color="white",
                          line_width=0.8, row=i + 1, col=v + 1)
        # All vars column
        cnt_a, _ = np.histogram(r_all, bins=bins, density=True) if len(r_all) > 0 \
                   else (np.zeros(Ne + 1), bins)
        fig.add_trace(go.Bar(x=list(range(Ne + 1)), y=cnt_a.tolist(),
                              name=name, legendgroup=name, showlegend=False,
                              marker_color=rgba, marker_line_color="#000",
                              marker_line_width=0.4),
                      row=i + 1, col=Nx + 1)
        fig.add_hline(y=uniform, line_dash="dash", line_color="white",
                      line_width=0.8, row=i + 1, col=Nx + 1)
    ens_label = "Forecast" if ens_key == "Xf_all" else "Analysis"
    fig.update_layout(
        template="cyberpunk",
        title=f"Rank Histograms ({ens_label}) — {_cfg_title(cfg)}" if cfg
              else f"Rank Histograms ({ens_label})",
        height=max(260 * n_r, 300),
        showlegend=True,
    )
    return fig


def fig_spaghetti_multi(results: dict, models: list, cfg=None,
                         cycle_range: list = None, var_idx: int = 0) -> go.Figure:
    active = [m for m in models if m in results and "ens_fcst_traj" in results[m]]
    if not active:
        return _make_empty_fig("No trajectory data — only baseline (b1) stores full trajectories")
    n_r = len(active)
    fig = make_subplots(rows=n_r, cols=1, row_titles=active,
                        shared_xaxes=True, vertical_spacing=0.04)
    for i, name in enumerate(active):
        res        = results[name]
        ens_traj   = res["ens_fcst_traj"]    # (T, Ne, M+1, Nx)
        truth_traj = res["truth_fcst_traj"]  # (T, M+1, Nx)
        T, Ne, Mp1, Nx = ens_traj.shape
        M       = Mp1 - 1
        k_start = int(cycle_range[0]) if cycle_range else 0
        k_end   = min(int(cycle_range[1]) if cycle_range else T, T)
        color   = NEON.get(name, "#888")
        thin    = _hex_rgba(color, 0.20)

        x_m, y_m, x_t, y_t, x_mu, y_mu = [], [], [], [], [], []
        for k in range(k_start, k_end):
            if np.any(np.isnan(ens_traj[k])):
                break
            t = (np.arange(Mp1) + k * M).tolist()
            for e in range(Ne):
                x_m  += t + [None]
                y_m  += ens_traj[k, e, :, var_idx].tolist() + [None]
            x_t  += t + [None]
            y_t  += truth_traj[k, :, var_idx].tolist() + [None]
            mu    = np.mean(ens_traj[k, :, :, var_idx], axis=0)
            x_mu += t + [None]
            y_mu += mu.tolist() + [None]

        fig.add_trace(go.Scattergl(x=x_m, y=y_m, mode="lines", name=f"{name} members",
                                    legendgroup=name, showlegend=(i == 0),
                                    line=dict(color=thin, width=0.5)), row=i + 1, col=1)
        fig.add_trace(go.Scattergl(x=x_mu, y=y_mu, mode="lines", name=f"{name} mean",
                                    legendgroup=name, showlegend=False,
                                    line=dict(color=color, width=2.0)), row=i + 1, col=1)
        #if i == 0:
        fig.add_trace(go.Scattergl(x=x_t, y=y_t, mode="lines", name="Truth",
                                    legendgroup="truth",
                                    line=dict(color="#ffffff", width=1.2)),
                          row=i + 1, col=1)
    k0 = cycle_range[0] if cycle_range else 0
    k1 = cycle_range[1] if cycle_range else "end"
    fig.update_layout(
        template="cyberpunk",
        title=f"Spaghetti — {VAR_NAMES[var_idx]} — cycles {k0}–{k1}",
        height=max(200 * n_r, 300),
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_xaxes(title_text="Forecast step (cumulative)", row=n_r, col=1)
    return fig


def fig_spread_reduction_multi(results: dict, models: list, cfg=None,
                                 cycle_range: list = None, var_idx: int = 0) -> go.Figure:
    active = [m for m in models if m in results and "ens_fcst_traj" in results[m]]
    if not active:
        return _make_empty_fig("No trajectory data — only baseline (b1) stores full trajectories")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=["Intra-cycle spread evolution",
                                        "Per-cycle spread: post-analysis (--) vs end-of-forecast (─)"],
                        vertical_spacing=0.14)
    max_valid = 0
    for name in active:
        res      = results[name]
        ens_traj = res["ens_fcst_traj"]
        T, Ne, Mp1, Nx = ens_traj.shape
        M       = Mp1 - 1
        k_start = int(cycle_range[0]) if cycle_range else 0
        k_end   = min(int(cycle_range[1]) if cycle_range else T, T)
        color   = NEON.get(name, "#888")

        a_sp, f_sp, valid_T = [], [], 0
        for k in range(T):
            if np.any(np.isnan(ens_traj[k])):
                break
            std_t = np.std(ens_traj[k, :, :, var_idx], axis=0)
            a_sp.append(float(std_t[0]))
            f_sp.append(float(std_t[-1]))
            valid_T += 1
        max_valid = max(max_valid, valid_T)

        # Top: NaN-separated intra-cycle curves
        x_top, y_top = [], []
        for k in range(k_start, min(k_end, valid_T)):
            t     = (np.arange(Mp1) + k * M).tolist()
            std_t = np.std(ens_traj[k, :, :, var_idx], axis=0)
            x_top += t + [None]
            y_top += std_t.tolist() + [None]
        fig.add_trace(go.Scatter(x=x_top, y=y_top, name=name, legendgroup=name,
                                  mode="lines", line=dict(color=_hex_rgba(color, 0.55), width=0.9)),
                      row=1, col=1)

        # Bottom: post-analysis (dashed) vs end-of-forecast (solid)
        cyc = list(range(valid_T))
        fig.add_trace(go.Scatter(x=cyc, y=a_sp, name=f"{name} post", legendgroup=name,
                                  showlegend=False,
                                  line=dict(color=color, dash="dash", width=1.0)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=cyc, y=f_sp, name=f"{name} end", legendgroup=name,
                                  showlegend=False,
                                  line=dict(color=color, width=1.0)),
                      row=2, col=1)

    fig.update_layout(
        template="cyberpunk",
        title=f"Spread Reduction — {VAR_NAMES[var_idx]}",
        legend=dict(orientation="h", y=-0.1),
    )
    fig.update_xaxes(title_text="Model step (cumulative)", row=1, col=1)
    fig.update_xaxes(title_text="DA Cycle", row=2, col=1)
    fig.update_yaxes(title_text=f"Std ({VAR_NAMES[var_idx]})")
    return fig


def fig_trajectory_comparison(results: dict, models: list, spinup: int = 50) -> go.Figure:
    T = max((len(r.get("xf_traj", [])) for r in results.values()), default=1)
    cycles = list(range(T))
    truth_src = "Lorenz63" if "Lorenz63" in results else (list(results.keys())[0] if results else None)
    if truth_src is None:
        return _make_empty_fig("No data")
    xt = results[truth_src]["xt_traj"]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_titles=VAR_NAMES, vertical_spacing=0.04)
    for vi in range(3):
        fig.add_trace(go.Scatter(
            x=cycles[spinup:], y=xt[spinup:, vi].tolist(),
            name="Truth", legendgroup="truth", showlegend=(vi == 0),
            line=dict(color="white", dash="dot", width=1.5)), row=vi + 1, col=1)
        for name in models:
            res = results.get(name)
            if res is None:
                continue
            color  = NEON.get(name, "#888")
            xf     = res["xf_traj"]
            valid  = ~np.isnan(xf[:, vi])
            c_plot = [cycles[j] for j in range(spinup, T) if valid[j]]
            v_plot = xf[spinup:, vi][valid[spinup:]].tolist()
            fig.add_trace(go.Scatter(
                x=c_plot, y=v_plot, name=name, legendgroup=name,
                showlegend=(vi == 0),
                line=dict(color=color, width=1.2,
                          dash="dash" if name == "Lorenz63" else "solid")),
                row=vi + 1, col=1)
    if spinup > 0:
        for vi in range(3):
            fig.add_vline(x=spinup, line_color="gray", line_dash="dot",
                          line_width=0.8, row=vi + 1, col=1)
    fig.update_layout(template="cyberpunk", title="Forecast Mean vs Truth",
                      legend=dict(orientation="h", y=-0.05))
    fig.update_xaxes(title_text="DA Cycle", row=3, col=1)
    return fig


def fig_sweep_rmse(pkl_dict: dict, bench: str, models: list) -> go.Figure:
    vals = SWEEP_VALS.get(bench)
    if not vals:
        return _make_empty_fig(f"No sweep data for {bench}")
    res_by_param = pkl_dict.get(bench, {}).get("results", {})
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Forecast RMSE", "Analysis RMSE"])
    for name in models:
        color  = NEON.get(name, "#888")
        f_rmse, a_rmse = [], []
        for v in vals:
            r  = res_by_param.get(v, {}).get(name, {})
            ef = float(np.nanmean(r["errorf"][DROP_T:])) if r and len(r.get("errorf", [])) > DROP_T else float("nan")
            ea = float(np.nanmean(r["errora"][DROP_T:])) if r and len(r.get("errora", [])) > DROP_T else float("nan")
            f_rmse.append(ef); a_rmse.append(ea)
        fig.add_trace(go.Scatter(x=vals, y=f_rmse, name=name, legendgroup=name,
                                  mode="lines+markers",
                                  line=dict(color=color,
                                            dash="dash" if name == "Lorenz63" else "solid",
                                            width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vals, y=a_rmse, name=name, legendgroup=name,
                                  showlegend=False, mode="lines+markers",
                                  line=dict(color=color,
                                            dash="dash" if name == "Lorenz63" else "solid",
                                            width=1.5)), row=1, col=2)
    xlabel = SWEEP_LABEL.get(bench, bench)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="RMSE", col=1)
    fig.update_layout(template="cyberpunk",
                      title=f"Sweep RMSE — {bench} ({xlabel})",
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ── Aggregated plot functions ──────────────────────────────────────

def fig_agg_boxplot(models: list, bench: str = "b1") -> go.Figure:
    if not _agg_cache.get("ready"):
        return _make_empty_fig("Aggregation cache loading — please wait")
    if bench != "b1":
        return _make_empty_fig("Boxplot available for baseline (b1) only")
    bd  = _agg_cache["b1"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Forecast RMSE", "Analysis RMSE"])
    for name in models:
        color = NEON.get(name, "#888")
        md    = bd.get(name, {})
        kw    = dict(marker_color=color, line_color=color, boxmean="sd",
                     fillcolor=_hex_rgba(color, 0.15))
        fig.add_trace(go.Box(y=md.get("errorf_seeds", []), name=name, **kw),
                      row=1, col=1)
        fig.add_trace(go.Box(y=md.get("errora_seeds", []), name=name, legendgroup=name,
                              showlegend=False, **kw), row=1, col=2)
    n = _agg_cache.get("files_loaded", "?")
    fig.update_layout(template="cyberpunk",
                      title=f"RMSE Distribution — baseline — {n} seeds")
    fig.update_yaxes(title_text="RMSE", col=1)
    return fig


def fig_agg_violin(models: list, bench: str = "b1") -> go.Figure:
    if not _agg_cache.get("ready"):
        return _make_empty_fig("Aggregation cache loading — please wait")
    if bench != "b1":
        return _make_empty_fig("Violin available for baseline (b1) only")
    bd  = _agg_cache["b1"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Forecast RMSE", "Analysis RMSE"])
    for name in models:
        color = NEON.get(name, "#888")
        md    = bd.get(name, {})
        kw    = dict(line_color=color, fillcolor=_hex_rgba(color, 0.15),
                     box_visible=True, meanline_visible=True, points="all",
                     pointpos=-1.5, marker=dict(color=color, size=4, opacity=0.5))
        fig.add_trace(go.Violin(y=md.get("errorf_seeds", []), name=name, **kw),
                      row=1, col=1)
        fig.add_trace(go.Violin(y=md.get("errora_seeds", []), name=name, legendgroup=name,
                                 showlegend=False, **kw), row=1, col=2)
    n = _agg_cache.get("files_loaded", "?")
    fig.update_layout(template="cyberpunk",
                      title=f"RMSE Violin — baseline — {n} seeds")
    return fig


def fig_agg_mean_std(models: list, bench: str = "b2") -> go.Figure:
    if not _agg_cache.get("ready"):
        return _make_empty_fig("Aggregation cache loading — please wait")
    if bench not in SWEEP_VALS:
        return _make_empty_fig("Mean±Std requires a sweep benchmark (b2–b5)")
    bd   = _agg_cache[bench]
    vals = SWEEP_VALS[bench]
    fig  = make_subplots(rows=1, cols=2, subplot_titles=["Forecast RMSE", "Analysis RMSE"])
    for name in models:
        color = NEON.get(name, "#888")
        rgba  = _hex_rgba(color, 0.15)
        dash  = "dash" if name == "Lorenz63" else "solid"
        fm, fs, am, as_ = [], [], [], []
        for v in vals:
            pv = bd.get(v, {}).get(name, {})
            f  = pv.get("errorf_seeds", [])
            a  = pv.get("errora_seeds", [])
            fm.append(float(np.nanmean(f)) if f else float("nan"))
            fs.append(float(np.nanstd(f))  if f else float("nan"))
            am.append(float(np.nanmean(a)) if a else float("nan"))
            as_.append(float(np.nanstd(a)) if a else float("nan"))
        fm, fs, am, as_ = np.array(fm), np.array(fs), np.array(am), np.array(as_)

        def _band(col, vals_x, mean, std, row):
            fig.add_trace(go.Scatter(
                x=vals_x + vals_x[::-1],
                y=(mean + std).tolist() + (mean - std).tolist()[::-1],
                fill="toself", fillcolor=rgba, showlegend=False,
                line=dict(color="rgba(0,0,0,0)"), legendgroup=name), row=row, col=col)

        fig.add_trace(go.Scatter(x=vals, y=fm.tolist(), name=name, legendgroup=name,
                                  mode="lines+markers",
                                  line=dict(color=color, dash=dash, width=1.5)),
                      row=1, col=1)
        _band(1, vals, fm, fs, 1)
        fig.add_trace(go.Scatter(x=vals, y=am.tolist(), name=name, legendgroup=name,
                                  showlegend=False, mode="lines+markers",
                                  line=dict(color=color, dash=dash, width=1.5)),
                      row=1, col=2)
        _band(2, vals, am, as_, 1)

    xlabel = SWEEP_LABEL.get(bench, bench)
    n = _agg_cache.get("files_loaded", "?")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="RMSE", col=1)
    fig.update_layout(template="cyberpunk",
                      title=f"Mean±Std RMSE — {bench} ({xlabel}) — {n} seeds",
                      legend=dict(orientation="h", y=-0.18))
    return fig


def fig_agg_sweep_line(models: list, bench: str = "b2") -> go.Figure:
    if not _agg_cache.get("ready"):
        return _make_empty_fig("Aggregation cache loading — please wait")
    if bench not in SWEEP_VALS:
        return _make_empty_fig("Sweep line requires b2–b5")
    bd   = _agg_cache[bench]
    vals = SWEEP_VALS[bench]
    fig  = make_subplots(rows=1, cols=2, subplot_titles=["Forecast RMSE", "Analysis RMSE"])
    for name in models:
        color = NEON.get(name, "#888")
        thin  = _hex_rgba(color, 0.14)
        dash  = "dash" if name == "Lorenz63" else "solid"
        all_f = [bd.get(v, {}).get(name, {}).get("errorf_seeds", []) for v in vals]
        all_a = [bd.get(v, {}).get(name, {}).get("errora_seeds", []) for v in vals]
        n_seeds = max((len(x) for x in all_f if x), default=0)
        for si in range(n_seeds):
            yf = [float(all_f[vi][si]) if si < len(all_f[vi]) else float("nan")
                  for vi in range(len(vals))]
            ya = [float(all_a[vi][si]) if si < len(all_a[vi]) else float("nan")
                  for vi in range(len(vals))]
            kw = dict(mode="lines", legendgroup=name, showlegend=False,
                      line=dict(color=thin, width=0.7))
            fig.add_trace(go.Scatter(x=vals, y=yf, **kw), row=1, col=1)
            fig.add_trace(go.Scatter(x=vals, y=ya, **kw), row=1, col=2)
        f_mean = [float(np.nanmean(all_f[vi])) if all_f[vi] else float("nan") for vi in range(len(vals))]
        a_mean = [float(np.nanmean(all_a[vi])) if all_a[vi] else float("nan") for vi in range(len(vals))]
        fig.add_trace(go.Scatter(x=vals, y=f_mean, name=name, legendgroup=name,
                                  mode="lines+markers",
                                  line=dict(color=color, dash=dash, width=2.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=vals, y=a_mean, name=name, legendgroup=name,
                                  showlegend=False, mode="lines+markers",
                                  line=dict(color=color, dash=dash, width=2.5)),
                      row=1, col=2)
    xlabel = SWEEP_LABEL.get(bench, bench)
    n = _agg_cache.get("files_loaded", "?")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="RMSE", col=1)
    fig.update_layout(template="cyberpunk",
                      title=f"Spaghetti+Mean RMSE — {bench} ({xlabel}) — {n} seeds",
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ─────────────────────────────────────────────────────────────────
# 5. Viewport factory
# ─────────────────────────────────────────────────────────────────

def _ctrl(label: str, child, flex: str = "1") -> html.Div:
    return html.Div([html.Div(label, style=S_LABEL), child],
                    style={"flex": flex, "minWidth": "80px"})


def make_viewport(vid: str) -> html.Div:
    fresh = [p.name for p in _discover_pkls()]
    pkl_opts   = [{"label": "◈ all seeds (agg)", "value": "__agg__"}] + \
                 [{"label": n, "value": n} for n in fresh]
    bench_opts = [
        {"label": "B1 — Baseline",      "value": "b1"},
        {"label": "B2 — Forecast M",    "value": "b2"},
        {"label": "B3 — Ensemble Ne",   "value": "b3"},
        {"label": "B4 — Obs fraction",  "value": "b4"},
        {"label": "B5 — Obs noise σ",   "value": "b5"},
    ]
    var_opts = [{"label": f"{v} (var {i})", "value": i} for i, v in enumerate(VAR_NAMES)]
    default_pkl = fresh[0] if fresh else None
    dd_style = {"backgroundColor": "#0a0a14", "color": _TEXT,
                "fontFamily": "Courier New, monospace"}

    return html.Div([
        # Row 1
        html.Div([
            _ctrl("PKL FILE", dcc.Dropdown(
                id={"type": "vp-pkl",   "index": vid},
                options=pkl_opts, value=default_pkl, clearable=False,
                style=dd_style), flex="2"),
            _ctrl("BENCHMARK", dcc.Dropdown(
                id={"type": "vp-bench", "index": vid},
                options=bench_opts, value="b1", clearable=False,
                style=dd_style)),
            _ctrl("PLOT TYPE", dcc.Dropdown(
                id={"type": "vp-plot",  "index": vid},
                options=PLOT_TYPES, value="rmse_comparison", clearable=False,
                style=dd_style)),
            _ctrl("PARAM VAL", dcc.Dropdown(
                id={"type": "vp-param", "index": vid},
                options=[{"label": "N/A", "value": "__all__"}],
                value="__all__", clearable=False,
                style=dd_style)),
            html.Button("✕", id={"type": "vp-close", "index": vid},
                        n_clicks=0, style=S_CLOSE),
        ], style=S_ROW),
        # Row 2
        html.Div([
            html.Div([
                html.Div("MODELS", style=S_LABEL),
                dcc.Checklist(
                    id={"type": "vp-models", "index": vid},
                    options=[{"label": f" {m}", "value": m} for m in MODELS],
                    value=MODELS[:], inline=True,
                    inputStyle={"marginRight": "3px", "accentColor": _GREEN},
                    labelStyle={"marginRight": "10px", "fontSize": "11px",
                                "color": _TEXT, "cursor": "pointer"},
                ),
            ], style={"flex": "3"}),
            _ctrl("STATE VAR", dcc.Dropdown(
                id={"type": "vp-var",   "index": vid},
                options=var_opts, value=0, clearable=False,
                style=dd_style)),
            html.Div([
                html.Div("CYCLE RANGE", style=S_LABEL),
                dcc.RangeSlider(
                    id={"type": "vp-cycles", "index": vid},
                    min=0, max=300, step=5, value=[0, 300],
                    marks={0: "0", 50: "50", 100: "100",
                           150: "150", 200: "200", 250: "250", 300: "300"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "2", "minWidth": "160px"}),
        ], style=S_ROW),
        # Graph
        dcc.Loading(type="circle", color=_GREEN, children=[
            dcc.Graph(
                id={"type": "vp-graph", "index": vid},
                style={"height": "600px"},
                figure=_make_empty_fig("Select options above to render plot"),
                config={"displayModeBar": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            ),
        ]),
    ], id={"type": "vp-card", "index": vid}, style=S_CARD)


# ─────────────────────────────────────────────────────────────────
# 6. App + layout
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="EnKF Dashboard // CYBER")

_sidebar = html.Div([
    html.H2("LSD DASHBOARD",
            style={"color": _GREEN, "fontSize": "14px", "letterSpacing": "3px",
                   "marginBottom": "2px", "textTransform": "uppercase"}),
    html.Div("// Lorenz Surrogates DataAssimilation mode active",
             style={"color": f"{_GREEN}55", "fontSize": "9px", "marginBottom": "10px"}),
    html.Hr(style=S_HR),
    html.Button("[ + ADD VIEWPORT ]",
                id="add-vp-btn", n_clicks=0,
                style={**S_BTN, "width": "100%", "marginBottom": "12px"}),
    html.Hr(style=S_HR),
    html.Div("SEED FILES", style=S_LABEL),
    html.Div(id="pkl-count", children=f"{len(PKL_FILES)} pkls found",
             style={"fontSize": "11px", "color": _TEXT, "marginBottom": "8px"}),
    html.Div("AGG CACHE", style=S_LABEL),
    html.Div(id="agg-status", children="Initializing…",
             style={"fontSize": "11px", "color": _TEXT, "marginBottom": "8px"}),
    dcc.Interval(id="agg-poll", interval=1000, n_intervals=0, disabled=False),
    html.Hr(style=S_HR),
    html.Div("USAGE", style=S_LABEL),
    html.Div([
        html.P("1. Click + ADD VIEWPORT",        style={"margin": "2px 0", "fontSize": "10px"}),
        html.P("2. Select pkl & benchmark",       style={"margin": "2px 0", "fontSize": "10px"}),
        html.P("3. Choose plot type",             style={"margin": "2px 0", "fontSize": "10px"}),
        html.P("4. Filter models & cycles",       style={"margin": "2px 0", "fontSize": "10px"}),
        html.P("5. Panels are independent ✓",
               style={"margin": "4px 0", "fontSize": "10px", "color": _GREEN}),
    ], style={"color": f"{_TEXT}99"}),
], style=S_SIDEBAR)

app.layout = html.Div([
    dcc.Store(id="vp-ids", data=[]),
    _sidebar,
    html.Div(id="vp-container", style=S_GRID),
], style=S_PAGE)

app.index_string = app.index_string.replace(
    "</head>", f"<style>{CYBERPUNK_CSS}</style></head>"
)

# ─────────────────────────────────────────────────────────────────
# 7. Callbacks
# ─────────────────────────────────────────────────────────────────

@callback(
    Output("vp-container", "children"),
    Output("vp-ids",       "data"),
    Input("add-vp-btn",                           "n_clicks"),
    Input({"type": "vp-close", "index": ALL},     "n_clicks"),
    State("vp-ids", "data"),
    prevent_initial_call=True,
)
def manage_viewports(add_clicks, close_clicks, store_ids):
    triggered = ctx.triggered_id

    if triggered == "add-vp-btn":
        new_vid = uuid.uuid4().hex[:12]
        patch = Patch()
        patch.append(make_viewport(new_vid))
        return patch, store_ids + [new_vid]

    if isinstance(triggered, dict) and triggered.get("type") == "vp-close":
        target = triggered["index"]
        # Guard: only close if this button was genuinely clicked (n_clicks > 0)
        trig_val = ctx.triggered[0].get("value", 0) if ctx.triggered else 0
        if not trig_val or target not in store_ids:
            return no_update, no_update
        pos = store_ids.index(target)
        patch = Patch()
        del patch[pos]
        return patch, [i for i in store_ids if i != target]

    return no_update, no_update


@callback(
    Output({"type": "vp-param", "index": MATCH}, "options"),
    Output({"type": "vp-param", "index": MATCH}, "value"),
    Input({"type": "vp-bench", "index": MATCH},  "value"),
    prevent_initial_call=True,
)
def update_param_opts(bench):
    if bench == "b1":
        return [{"label": "N/A", "value": "__all__"}], "__all__"
    vals  = SWEEP_VALS.get(bench, [])
    label = SWEEP_LABEL.get(bench, bench).split("(")[0].strip()
    opts  = [{"label": "All (sweep overview)", "value": "__all__"}] + \
            [{"label": f"{label} = {v}", "value": v} for v in vals]
    return opts, (vals[0] if vals else "__all__")


@callback(
    Output("agg-status", "children"),
    Output("agg-poll",   "disabled"),
    Input("agg-poll",    "n_intervals"),
)
def poll_agg(n):
    with _cache_lock:
        ready  = _agg_cache.get("ready", False)
        loaded = _agg_cache.get("files_loaded", 0)
        total  = _agg_cache.get("total", "?")
    if ready:
        return [html.Span("✓ ", style={"color": _GREEN}),
                html.Span(f"{loaded} seeds ready", style={"color": _TEXT})], True
    return html.Span(f"Loading… {loaded}/{total}", style={"color": "#ffff00"}), False


@callback(
    Output({"type": "vp-graph",  "index": MATCH}, "figure"),
    Input({"type": "vp-pkl",     "index": MATCH}, "value"),
    Input({"type": "vp-bench",   "index": MATCH}, "value"),
    Input({"type": "vp-plot",    "index": MATCH}, "value"),
    Input({"type": "vp-param",   "index": MATCH}, "value"),
    Input({"type": "vp-models",  "index": MATCH}, "value"),
    Input({"type": "vp-cycles",  "index": MATCH}, "value"),
    Input({"type": "vp-var",     "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def render_graph(pkl_file, bench, plot_type, param_val, models,
                  cycle_range, var_idx):
    if not models:
        return _make_empty_fig("No models selected")
    if pkl_file is None or plot_type is None or bench is None:
        return _make_empty_fig("Select options above")

    var_idx     = int(var_idx) if var_idx is not None else 0
    cycle_range = list(cycle_range) if cycle_range else [0, 300]

    # ── Aggregation mode ──────────────────────────────────────────
    if pkl_file == "__agg__":
        if   plot_type == "agg_boxplot":    return fig_agg_boxplot(models, bench)
        elif plot_type == "agg_violin":     return fig_agg_violin(models, bench)
        elif plot_type == "agg_mean_std":   return fig_agg_mean_std(models, bench)
        elif plot_type == "agg_sweep_line": return fig_agg_sweep_line(models, bench)
        else:
            return _make_empty_fig(
                "Select an  Agg:  plot type for aggregated mode\n"
                "(Agg: Boxplot / Violin / Mean±Std / Spaghetti+Mean)"
            )

    # ── Single-file mode ──────────────────────────────────────────
    try:
        pkl = get_pkl_data(pkl_file)
    except Exception as exc:
        return _make_empty_fig(f"Error loading file:\n{exc}")

    if bench == "b1":
        results = pkl.get("b1", {})
        cfg     = None
    else:
        bench_data   = pkl.get(bench, {})
        results_dict = bench_data.get("results", {})
        setup        = bench_data.get("setup",   {})

        if param_val == "__all__":
            if plot_type == "sweep_rmse":
                return fig_sweep_rmse(pkl, bench, models)
            return _make_empty_fig(
                "Select a specific parameter value for single-cycle plots,\n"
                "or choose  'Sweep: RMSE'  for a sweep overview."
            )

        # Normalise param_val type to match pkl key type
        if bench in ("b2", "b3"):
            try:
                param_val = int(float(param_val))
            except (ValueError, TypeError):
                pass
        else:
            try:
                param_val = float(param_val)
            except (ValueError, TypeError):
                pass

        results = results_dict.get(param_val, {})
        cfg     = setup.get(param_val, None)

    if not results:
        return _make_empty_fig("No data found for this selection")

    # ── Dispatch ──────────────────────────────────────────────────
    if   plot_type == "rmse_comparison":
        return fig_rmse_comparison(results, models, cfg)
    elif plot_type == "spread_comparison":
        return fig_spread_comparison(results, models, cfg)
    elif plot_type == "spread_vs_rmse":
        return fig_spread_vs_rmse(results, models)
    elif plot_type == "talagrand_grid":
        return fig_talagrand_grid(results, models, cfg)
    elif plot_type == "spaghetti_multi":
        return fig_spaghetti_multi(results, models, cfg, cycle_range, var_idx)
    elif plot_type == "spread_reduction_multi":
        return fig_spread_reduction_multi(results, models, cfg, cycle_range, var_idx)
    elif plot_type == "trajectory_comparison":
        return fig_trajectory_comparison(results, models)
    elif plot_type == "sweep_rmse":
        if bench == "b1":
            return _make_empty_fig("Sweep: RMSE requires a sweep benchmark (b2–b5)")
        return fig_sweep_rmse(pkl, bench, models)
    elif plot_type in ("agg_boxplot", "agg_violin", "agg_mean_std", "agg_sweep_line"):
        return _make_empty_fig(
            "Select  '◈ all seeds (agg)'  in the PKL selector\n"
            "to use aggregated plot types."
        )
    return _make_empty_fig(f"Unknown plot type: {plot_type}")


# ─────────────────────────────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=8060, debug=True)
