from fastai.vision.all import *
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import numpy as np
import cv2
import io


# Multi-video viewer with global + per-video sliders (NumPy frames)
# ---------------------------------------------------------------
# - frames_list: list of np.ndarray videos shaped (T,H,W) or (T,H,W,3)
# - Supports uint8 or float [0,1], RGB/BGR, different lengths
# - Global slider moves all; per-video sliders can be locked/unlocked
# - Optional per-video offsets to align clips to the same “moment”
#
# Usage:
#   viewer = create_multi_video_viewer(
#       [vid1, vid2, vid3],
#       fps_list=[30, 60, 30],          # optional (for time readouts)
#       from_bgr=[True, True, False],   # set True if your frames are OpenCV BGR
#       offsets=[0, -12, +5],           # optional per-video index offsets
#   )
#
# Notes:
# - The global slider spans the LONGEST clip; shorter clips clamp to their last frame.
# - When "Lock to global" is on, that video’s local slider is disabled and follows global.

import numpy as np
import cv2
import ipywidgets as widgets
from IPython.display import display
from functools import lru_cache
from typing import Sequence, Optional, Union, List

def _ensure_uint8_image(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            amin, amax = float(arr.min()), float(arr.max())
            if amax > amin:
                arr = ((arr - amin) / (amax - amin) * 255.0).round().astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
    return arr

def _to_png_bytes(frame: np.ndarray, from_bgr: bool) -> bytes:
    img = _ensure_uint8_image(frame)
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        bgr = img if from_bgr else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Frame must be HxW or HxWx3.")
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed.")
    return buf.tobytes()

def create_multi_video_viewer(
    frames_list: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    fps_list: Optional[Sequence[Optional[float]]] = None,
    width: Union[int, Sequence[int]] = 384,
    from_bgr: Union[bool, Sequence[bool]] = False,
    offsets: Optional[Sequence[int]] = None,
    lock_defaults: Union[bool, Sequence[bool]] = True,
):
    n = len(frames_list)
    if n == 0:
        raise ValueError("frames_list must contain at least one video.")

    def _as_list(x, n, default):
        if isinstance(x, (list, tuple)):
            if len(x) != n:
                raise ValueError(f"Expected len {n}, got {len(x)}.")
            return list(x)
        return [x if x is not None else default for _ in range(n)]

    titles = titles or [f"Video {i+1}" for i in range(n)]
    fps_list = _as_list(fps_list if fps_list is not None else [None]*n, n, None)
    width_list = _as_list(width, n, 384)
    from_bgr_list = _as_list(from_bgr, n, False)
    lock_defaults = _as_list(lock_defaults, n, True)
    offsets = _as_list(offsets if offsets is not None else [0]*n, n, 0)

    lens, shapes = [], []
    for i, vid in enumerate(frames_list):
        if not isinstance(vid, np.ndarray) or vid.ndim not in (3, 4):
            raise ValueError(f"frames_list[{i}] must be (T,H,W) or (T,H,W,3) array.")
        lens.append(vid.shape[0]); shapes.append(vid.shape)

    max_len = max(lens)

    # Cache PNG encodes for snappy scrubbing
    @lru_cache(maxsize=1024)
    def _enc(video_idx: int, frame_idx: int) -> bytes:
        frame_idx = int(np.clip(frame_idx, 0, lens[video_idx]-1))
        return _to_png_bytes(frames_list[video_idx][frame_idx], from_bgr_list[video_idx])

    # Build widgets
    global_slider = widgets.IntSlider(
        description="Global",
        min=0, max=max_len-1, step=1, value=0,
        continuous_update=True, readout=True, readout_format="d",
        layout=widgets.Layout(width="95%"),
    )

    image_widgets: List[widgets.Image] = []
    local_sliders: List[widgets.IntSlider] = []
    lock_boxes: List[widgets.Checkbox] = []
    readouts: List[widgets.Label] = []
    titles_html: List[widgets.HTML] = []

    def _update_one(i: int, idx: int):
        clamped = int(np.clip(idx, 0, lens[i]-1))
        image_widgets[i].value = _enc(i, clamped)
        label = f"f {clamped}/{lens[i]-1}"
        fps = fps_list[i]
        if isinstance(fps, (int, float)) and fps > 0:
            label += f"   t {clamped/float(fps):.3f}s"
        readouts[i].value = label

    panels = []
    for i in range(n):
        img = widgets.Image(value=_enc(i, 0), format="png", width=width_list[i])
        image_widgets.append(img)

        local = widgets.IntSlider(
            description="Frame",
            min=0, max=lens[i]-1, step=1, value=0,
            continuous_update=True, readout=True, readout_format="d",
            layout=widgets.Layout(width="95%"),
        )
        local_sliders.append(local)

        lock = widgets.Checkbox(value=lock_defaults[i], description="Lock to global", indent=False)
        lock_boxes.append(lock)

        readout = widgets.Label()
        readouts.append(readout)
        _update_one(i, 0)

        titles_html.append(widgets.HTML(f"<b>{titles[i]}</b> "
                                        f"<span style='font-size:11px;color:gray;'>{shapes[i]}</span>"))

        panels.append(
            widgets.VBox(
                [titles_html[i],
                 img,
                 widgets.HBox([local, lock], layout=widgets.Layout(align_items="center")),
                 readout],
                layout=widgets.Layout(padding="4px", border="1px solid #3333", border_radius="10px"),
            )
        )

    ui = widgets.VBox([
        widgets.HBox([global_slider], layout=widgets.Layout(padding="2px")),
        widgets.HBox(panels, layout=widgets.Layout(justify_content="space-between"))
    ])

    # Wire interactions with a guard to prevent feedback loops
    updating = {"on": False}

    def on_global(change):
        if updating["on"]: return
        updating["on"] = True
        g = int(change["new"])
        for i in range(n):
            if lock_boxes[i].value:
                tgt = int(g + offsets[i])
                local_sliders[i].value = int(np.clip(tgt, 0, lens[i]-1))
                local_sliders[i].disabled = True
                _update_one(i, tgt)
        updating["on"] = False

    def make_local_handler(i: int):
        def _h(change):
            if updating["on"]: return
            if lock_boxes[i].value:  # locked -> ignore (slider is disabled anyway)
                return
            updating["on"] = True
            _update_one(i, int(change["new"]))
            updating["on"] = False
        return _h

    def make_lock_handler(i: int):
        def _h(change):
            locked = bool(change["new"])
            local_sliders[i].disabled = locked
            if locked:
                tgt = int(global_slider.value + offsets[i])
                local_sliders[i].value = int(np.clip(tgt, 0, lens[i]-1))
                _update_one(i, tgt)
        return _h

    global_slider.observe(on_global, names="value")
    for i in range(n):
        local_sliders[i].observe(make_local_handler(i), names="value")
        lock_boxes[i].observe(make_lock_handler(i), names="value")
        local_sliders[i].disabled = bool(lock_boxes[i].value)  # initial state

    display(ui)
    return {
        "ui": ui,
        "global_slider": global_slider,
        "image_widgets": image_widgets,
        "local_sliders": local_sliders,
        "lock_checkboxes": lock_boxes,
        "readouts": readouts,
    }



def get_frame_plot(kps, plot=False, print_output=False):
    m = compute_right_arm_metrics_v2(kps, conf_thresh=0.5, smooth_win=7)
    
    best_idx, details = pick_straightest_frame_v2(
        m,
        require_wrist_above='both',   # ← your requirement: wrist above elbow AND shoulder
        peak_window=10,
        avoid_last_n=3,
        prefer_local_peaks=True
    )
    if print_output:
        print(best_idx, details)
    if plot:
        fig = plot_angle_series_with_window(m, best_idx, peak_idx=details['peak_idx'], win=details['window'])
        _   = plot_skeleton(kps, best_idx, metrics=m, mode='triptych')
    return best_idx, details



import numpy as np
import matplotlib.pyplot as plt

# --- COCO-17 indices (update if your model differs)
COCO = dict(LSH=5, RSH=6, LEL=7, REL=8, LWR=9, RWR=10)

# --- Helpers ---------------------------------------------------------

def _get_xyc(kps):
    """Return (xy, conf_or_None) with shapes (T,17,2) and (T,17) or None."""
    kps = np.asarray(kps)
    if kps.ndim != 3 or kps.shape[1] < 11 or kps.shape[2] < 2:
        raise ValueError(f"Expect kps shaped (T,17,2|3); got {kps.shape}")
    xy = kps[..., :2].astype(np.float64)
    conf = kps[..., 2] if kps.shape[2] >= 3 else None
    return xy, conf

def _nanma(x, w):
    """NaN-aware moving average (same length)."""
    if w <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    m = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    k = np.ones(w, float)
    num = np.convolve(x0, k, mode='same')
    den = np.convolve(m,  k, mode='same')
    out = num / np.where(den == 0, np.nan, den)
    return out

def _angle_at_elbow(S, E, W, eps=1e-8):
    """Angle at elbow (deg). 180 = straight."""
    ES = S - E
    EW = W - E
    d  = np.einsum('...i,...i', ES, EW)
    n  = (np.linalg.norm(ES, axis=-1) * np.linalg.norm(EW, axis=-1)) + eps
    cos = np.clip(d / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos)), cos

def _elbow_offline_px(S, E, W, eps=1e-8):
    """Perpendicular distance of E from line SW (pixels)."""
    SW = W - S
    SE = E - S
    num = np.abs(SW[...,0]*SE[...,1] - SW[...,1]*SE[...,0])
    den = np.linalg.norm(SW, axis=-1) + eps
    return num / den

def _extension_ratio(S, E, W, eps=1e-8):
    """|SW| / (|SE| + |EW|); -> 1 when straight."""
    SW = np.linalg.norm(W - S, axis=-1)
    SE = np.linalg.norm(E - S, axis=-1)
    EW = np.linalg.norm(W - E, axis=-1)
    return SW / (SE + EW + eps)

def _shoulder_width(xy):
    """Per-frame right-left shoulder distance; NaN if missing."""
    has_L = np.all(np.isfinite(xy[:, COCO['LSH']]), axis=-1)
    has_R = np.all(np.isfinite(xy[:, COCO['RSH']]), axis=-1)
    d = np.full((xy.shape[0],), np.nan)
    mask = has_L & has_R
    if np.any(mask):
        d[mask] = np.linalg.norm(
            xy[mask, COCO['RSH']] - xy[mask, COCO['LSH']], axis=-1
        )
    return d

def _local_maxima(y):
    """Indices of simple local maxima (strict), ignoring NaNs."""
    y = np.asarray(y, float)
    idx = []
    for i in range(1, len(y)-1):
        if not np.isfinite(y[i]): 
            continue
        yl, yc, yr = y[i-1], y[i], y[i+1]
        if np.isfinite(yl) and np.isfinite(yr) and yc >= yl and yc >= yr:
            idx.append(i)
    return np.array(idx, dtype=int)

# --- Metrics (v2): base validity first, wrist-above used later ----------

def compute_right_arm_metrics_v2(kps, conf_thresh=0.4, smooth_win=5, min_arm_len_px=10.0):
    """
    Compute per-frame metrics for the RIGHT arm; do NOT bake in 'wrist-above' to validity.
    """
    xy, conf = _get_xyc(kps)
    T = xy.shape[0]
    S = xy[:, COCO['RSH']]
    E = xy[:, COCO['REL']]
    W = xy[:, COCO['RWR']]

    angle_deg, _ = _angle_at_elbow(S, E, W)
    offline_px   = _elbow_offline_px(S, E, W)
    extension    = _extension_ratio(S, E, W)
    shw          = _shoulder_width(xy)
    offline_norm = offline_px / shw

    # Base gating: confidence + minimal lengths + finite metrics
    if conf is not None:
        conf_S = conf[:, COCO['RSH']]
        conf_E = conf[:, COCO['REL']]
        conf_W = conf[:, COCO['RWR']]
        mean_conf = np.nanmean(np.stack([conf_S, conf_E, conf_W], axis=-1), axis=-1)
        conf_ok   = (conf_S >= conf_thresh) & (conf_E >= conf_thresh) & (conf_W >= conf_thresh)
    else:
        mean_conf = np.ones(T)
        conf_ok   = np.ones(T, dtype=bool)

    ES_len = np.linalg.norm(S - E, axis=-1)
    EW_len = np.linalg.norm(W - E, axis=-1)
    len_ok = (ES_len > min_arm_len_px) & (EW_len > min_arm_len_px)
    finite_ok = np.isfinite(angle_deg) & np.isfinite(extension) & np.isfinite(offline_norm)

    valid_base = conf_ok & len_ok & finite_ok

    # Smoothed signals (using base validity)
    angle_s = _nanma(np.where(valid_base, angle_deg, np.nan), smooth_win)
    ext_s   = _nanma(np.where(valid_base, extension, np.nan), smooth_win)
    off_s   = _nanma(np.where(valid_base, offline_norm, np.nan), smooth_win)

    # Scores
    angle_score = np.clip((angle_s - 120.0) / 60.0, 0.0, 1.0)
    ext_score   = np.clip(ext_s, 0.0, 1.0)
    off_score   = 1.0 / (1.0 + 10.0 * np.maximum(off_s, 0.0))
    w_angle, w_ext, w_off = 0.55, 0.30, 0.15
    score_base = (w_angle*angle_score + w_ext*ext_score + w_off*off_score) * np.clip(mean_conf, 0, 1)
    score_base[~valid_base] = np.nan

    # Wrist-above masks (OpenCV coordinates: smaller y = higher)
    wrist_above_elbow = (W[:,1] < E[:,1])
    wrist_above_both  = wrist_above_elbow & (W[:,1] < S[:,1])

    return dict(
        angle_deg=angle_deg, angle_deg_s=angle_s,
        extension=extension, extension_s=ext_s,
        offline_norm=offline_norm, offline_norm_s=off_s,
        angle_score=angle_score, ext_score=ext_score, off_score=off_score,
        score_base=score_base, valid_base=valid_base,
        mean_conf=mean_conf, ES_len=ES_len, EW_len=EW_len, shw=shw,
        wrist_above_elbow=wrist_above_elbow, wrist_above_both=wrist_above_both
    )

# --- Picker (v2): peak-of-angle -> window -> wrist-above ----------------

def pick_straightest_frame_v2(
    m,
    require_wrist_above='both',   # {'both','elbow','none'}
    peak_window=10,               # +/- frames around the angle peak
    avoid_last_n=3,               # mild tail penalty
    prefer_local_peaks=True,
    angle_margin_deg=1.0
):
    angle_s   = m['angle_deg_s']
    angle_raw = m['angle_deg']
    score     = m['score_base']
    valid0    = m['valid_base']

    # 1) angle peak using base-valid
    ang = np.where(valid0, angle_s, np.nan)
    if prefer_local_peaks:
        cands = _local_maxima(ang)
        peak_idx = int(cands[np.nanargmax(ang[cands])]) if cands.size else int(np.nanargmax(ang))
    else:
        peak_idx = int(np.nanargmax(ang))
    if not np.isfinite(ang[peak_idx]):
        return None, dict(reason="No base-valid frames")

    # 2) candidate window
    lo = max(0, peak_idx - peak_window); hi = min(len(ang)-1, peak_idx + peak_window)
    in_win = np.zeros_like(valid0, bool); in_win[lo:hi+1] = True

    # 3) wrist-above masks
    if require_wrist_above == 'both':
        above = m['wrist_above_both']
    elif require_wrist_above == 'elbow':
        above = m['wrist_above_elbow']
    else:
        above = np.ones_like(valid0, bool)

    cand = valid0 & in_win & above & np.isfinite(score)

    # 4) relax if empty
    if not np.any(cand):
        if require_wrist_above == 'both':
            cand = valid0 & in_win & m['wrist_above_elbow'] & np.isfinite(score)
        if not np.any(cand):
            cand = valid0 & in_win & np.isfinite(score)
    if not np.any(cand):
        cand = valid0 & above & np.isfinite(score)
    if not np.any(cand):
        cand = valid0 & np.isfinite(score)
    if not np.any(cand):
        return None, dict(reason="No valid candidate after all relaxations")

    cand_idx = np.where(cand)[0]

    # 5) mild tail penalty
    if avoid_last_n > 0:
        tail = np.zeros_like(score, float); tail[-avoid_last_n:] = 0.03
        score_eff = score - tail
    else:
        score_eff = score

    # 6) rank: (a) max raw angle (within margin), (b) highest score_eff,
    #           (c) smaller offline_norm_s, (d) higher extension_s, (e) higher conf, (f) closer to peak
    angle_c = angle_raw[cand_idx]
    max_ang = np.nanmax(angle_c)
    close_by_angle = cand_idx[np.where(np.abs(angle_c - max_ang) <= angle_margin_deg)[0]]
    if close_by_angle.size == 0:
        close_by_angle = cand_idx

    off_s = m['offline_norm_s']; ext_s = m['extension_s']; conf = m['mean_conf']
    def _key(i):
        return (
            - (angle_raw[i] if np.isfinite(angle_raw[i]) else -np.inf),
            - (score_eff[i]  if np.isfinite(score_eff[i])  else -np.inf),
            + (off_s[i]      if np.isfinite(off_s[i])      else np.inf),
            - (ext_s[i]      if np.isfinite(ext_s[i])      else -np.inf),
            - (conf[i]       if np.isfinite(conf[i])       else -np.inf),
            abs(i - peak_idx)
        )
    best_idx = min(close_by_angle, key=_key)

    details = dict(
        best_idx=int(best_idx),
        peak_idx=int(peak_idx),
        window=[int(lo), int(hi)],
        policy=f"angle-peak-window wrist={require_wrist_above}",
        best_angle_deg=float(m['angle_deg'][best_idx]),
        best_extension=float(m['extension'][best_idx]),
        best_offline_norm=float(m['offline_norm'][best_idx]),
        best_score=float(m['score_base'][best_idx]),
        wrist_above_both=bool(m['wrist_above_both'][best_idx]),
        wrist_above_elbow=bool(m['wrist_above_elbow'][best_idx]),
        base_valid=bool(m['valid_base'][best_idx])
    )
    return int(best_idx), details

# --- Plots -------------------------------------------------------------

def plot_angle_series_with_window(m, best_idx=None, title="Right arm straightness (angle & score)", peak_idx=None, win=None):
    T = m['angle_deg'].shape[0]
    x = np.arange(T)
    ang = m['angle_deg_s']
    scr = m['score_base']
    base_valid = m['valid_base']

    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_title(title)
    ax1.plot(x, ang, lw=1.8, label='Elbow angle (deg, smoothed)')
    if np.any(~base_valid):
        ax1.fill_between(x, np.nanmin(ang)-1, np.nanmax(ang)+1, where=~base_valid, alpha=0.1, label='invalid (base)')
    if win is not None:
        lo, hi = win
        ax1.axvspan(lo, hi, color='k', alpha=0.06, label='peak window')
    if peak_idx is not None:
        ax1.axvline(peak_idx, ls='--', lw=1.2, label='angle peak')
    if best_idx is not None:
        ax1.axvline(best_idx, color='tab:green', lw=2, label='chosen')

    ax1.set_ylabel('Angle (deg)'); ax1.set_xlabel('Frame')

    ax2 = ax1.twinx()
    ax2.plot(x, scr, lw=1.0, alpha=0.85, label='Combined score')
    ax2.set_ylabel('Score (0-1)')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2= ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='lower right')
    plt.tight_layout()
    return fig

def plot_skeleton(
    kps, idx, metrics=None, ax=None,
    mode='overlay',
    layout='horizontal',          # 'horizontal' (default) or 'vertical' for triptych
    title_style='two-line',       # 'two-line' (default), 'single', or 'box'
    title_fontsize=10
):
    """
    Plot right-arm skeleton for frame idx plus neighbors (idx-1, idx+1).
    Titles include R/L angles; if `metrics` is provided, also show score/offline/ext/above/valid.

    - layout='vertical' stacks panels to avoid title overlap.
    - title_style='two-line' splits long titles across two lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- helpers assumed available: _get_xyc, _angle_at_elbow, COCO ---
    xy, _ = _get_xyc(kps)
    T = xy.shape[0]
    idx = int(np.clip(idx, 0, T-1))

    frames = [max(0, idx-1), idx, min(T-1, idx+1)]
    labels = ['t-1', 't', 't+1']
    uniq = []
    for f, l in zip(frames, labels):
        if not uniq or uniq[-1][0] != f:
            uniq.append((f, l))

    def _elbow_angle_one(side, f):
        if side == 'R':
            S = xy[f, COCO['RSH']]; E = xy[f, COCO['REL']]; W = xy[f, COCO['RWR']]
        else:
            S = xy[f, COCO['LSH']]; E = xy[f, COCO['LEL']]; W = xy[f, COCO['LWR']]
        a, _ = _angle_at_elbow(S[None], E[None], W[None])
        a0 = a[0] if a.size else np.nan
        return float(a0) if np.isfinite(a0) else np.nan

    def _fmt(x, nd=3):
        return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    # Build per-panel info with separate lines for multi-line titles
    panels = []
    for f, l in uniq:
        aR = _elbow_angle_one('R', f)
        aL = _elbow_angle_one('L', f)
        line1 = f"{l} (f={f}) | R {_fmt(aR,1)}° | L {_fmt(aL,1)}°"
        if metrics is not None:
            sc  = metrics['score_base'][f]
            off = metrics['offline_norm'][f]
            ext = metrics['extension'][f]
            ab  = bool(metrics['wrist_above_both'][f]) if 'wrist_above_both' in metrics else None
            vb  = bool(metrics['valid_base'][f]) if 'valid_base' in metrics else None
            line2 = f"score {_fmt(sc)} | off {_fmt(off,4)} | ext {_fmt(ext,3)}"
            if ab is not None: line2 += f" | above {ab}"
            if vb is not None: line2 += f" | valid {vb}"
        else:
            line2 = ""  # no metrics
        panels.append((f, line1, line2))

    def _draw_right(ax, f, color=None, alpha=1.0, label=None):
        S = xy[f, COCO['RSH']]; E = xy[f, COCO['REL']]; W = xy[f, COCO['RWR']]
        ax.plot([S[0],E[0]],[S[1],E[1]], marker='o', lw=2, color=color, alpha=alpha)
        ax.plot([E[0],W[0]],[E[1],W[1]], marker='o', lw=2, color=color, alpha=alpha)
        ax.scatter([S[0],E[0],W[0]],[S[1],E[1],W[1]], s=36, color=color, alpha=alpha, label=label)

    # ----- overlay mode (single axes) -----
    if mode == 'overlay':
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
        palette = ['tab:orange','tab:blue','tab:green']; alphas=[0.35,1.0,0.35]
        for (f, line1, line2), c, a in zip(panels, palette[:len(panels)], alphas[:len(panels)]):
            _draw_right(ax, f, color=c, alpha=a, label=line1.split(" | ")[0])  # short label
        if title_style == 'two-line' and panels:
            l1 = " | ".join([p[1] for p in panels])
            l2 = " | ".join([p[2] for p in panels if p[2]])
            title = l1 + ("\n" + l2 if l2 else "")
        elif title_style == 'box' and panels:
            title = ""
            box_text = "\n".join([p[1] + ("\n" + p[2] if p[2] else "") for p in panels])
            ax.text(0.01, 0.99, box_text, va='top', ha='left',
                    transform=ax.transAxes, fontsize=title_fontsize,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0.5))
        else:  # single-line
            title = " | ".join([p[1] + (" | " + p[2] if p[2] else "") for p in panels])

        if title_style != 'box':
            ax.set_title(title, fontsize=title_fontsize, loc='left', wrap=True)

        ax.invert_yaxis(); ax.set_aspect('equal'); ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')
        return ax

    # ----- triptych mode (multiple subplots) -----
    elif mode == 'triptych':
        if layout == 'vertical':
            fig, axs = plt.subplots(len(panels), 1, figsize=(6, 4.8*len(panels)), constrained_layout=True)
        else:
            fig, axs = plt.subplots(1, len(panels), figsize=(6*len(panels), 4.8), constrained_layout=True)
        if len(panels) == 1:
            axs = [axs]

        for ax_i, (f, line1, line2) in zip(axs, panels):
            _draw_right(ax_i, f)
            if title_style == 'box':
                ax_i.set_title("")  # put content in an anchored box inside the axes
                txt = line1 + ("\n" + line2 if line2 else "")
                ax_i.text(0.01, 0.99, txt, va='top', ha='left',
                          transform=ax_i.transAxes, fontsize=title_fontsize,
                          bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.5))
            elif title_style == 'two-line':
                ax_i.set_title(line1 + ("\n" + line2 if line2 else ""),
                               fontsize=title_fontsize, loc='left', wrap=True)
            else:
                ax_i.set_title(line1 + (" | " + line2 if line2 else ""),
                               fontsize=title_fontsize, loc='left', wrap=True)
            ax_i.invert_yaxis(); ax_i.set_aspect('equal'); ax_i.set_xlabel('x'); ax_i.set_ylabel('y')

        # Add a compact suptitle and let constrained_layout handle spacing
        fig.suptitle(f"Right arm around frame {idx}", y=0.995, fontsize=title_fontsize+1)
        return fig

    else:
        raise ValueError("mode must be 'overlay' or 'triptych'")