from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy as np
import logging
import matplotlib.pyplot as plt
import textwrap

# ------------------------------
# COCO keypoint indices (17-pt)
# ------------------------------
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16

# ------------------------------
# Logger (informative but quiet)
# ------------------------------
logger = logging.getLogger("plots_only")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ------------------------------
# Keypoint configuration dataclass
# ------------------------------
@dataclass
class KeypointConfig:
    """Configuration for a single keypoint to track and plot."""
    index: int                    # COCO keypoint index (0-16)
    label: str                    # Display label for plots
    plot_y: bool = True          # Whether to plot y-coordinate
    plot_x: bool = False         # Whether to plot x-coordinate
    color: Optional[str] = None  # Optional color for this keypoint's line
    linestyle: str = "-"         # Line style: "-", "--", ":", "-."
    alpha: float = 1.0           # Transparency (0-1)


@dataclass
class AngleConfig:
    """Configuration for an angle calculation between keypoints."""
    name: str                    # Name for the angle (e.g., "Shoulder angle")
    point1_idx: int             # First keypoint index
    point2_idx: int             # Second keypoint index
    point3_idx: Optional[int] = None    # Third keypoint index
    show_delta: bool = False    # Whether to show delta from baseline
    delta_baseline_n: int = 5   # Frames to use for baseline calculation


# ------------------------------
# Parameters for plotting
# ------------------------------
@dataclass
class UpperPlotParams:
    handedness: str = "right"        # 'right' or 'left' (just affects which wrist is called "lead")
    fps: Optional[float] = 60.0      # used for an optional seconds axis; set None to hide seconds axis
    min_conf: float = 0.35           # confidences below this are filled from neighbors before smoothing
    smooth_win: int = 9              # odd moving-average window (frames)
    invert_y: bool = True            # typical for image coords: smaller y == higher in image
    show_seconds_axis: bool = True   # add a top x-axis in seconds if fps is provided
    shoulder_delta_baseline_n: int = 5  # use the median of the first N frames as baseline for Î”Â° (robust)
    show_shoulder_delta: bool = True # plot Î”Â° from early-frame baseline alongside absolute shoulder angle

    # NEW: Flexible keypoint tracking (up to 10 keypoints)
    keypoints: List[KeypointConfig] = field(default_factory=list)

    # NEW: Flexible angle tracking
    angles: List[AngleConfig] = field(default_factory=list)

    # Legacy compatibility mode (if True, uses original wrist+shoulder behavior)
    use_legacy_mode: bool = True


# ------------------------------
# Small helpers
# ------------------------------
def _lead_trail_idxs(handedness: str) -> Dict[str, int]:
    """Define which side is 'lead' vs 'trail' given handedness."""
    h = handedness.lower()
    if h.startswith("r"):
        return dict(lead_sh=L_SH, lead_el=L_EL, lead_wr=L_WR, trail_sh=R_SH, trail_el=R_EL, trail_wr=R_WR)
    else:
        return dict(lead_sh=R_SH, lead_el=R_EL, lead_wr=R_WR, trail_sh=L_SH, trail_el=L_EL, trail_wr=L_WR)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Odd-length centered moving-average. Handles 1D or 2D arrays (time-major).
    For 2D: smooth each column independently.
    """
    win = max(1, int(win) | 1)  # force odd, >=1
    if x.ndim == 1:
        pad = win // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(win, dtype=float) / win
        return np.convolve(xp, ker, mode="valid")
    elif x.ndim == 2:
        out = np.empty_like(x, dtype=float)
        for d in range(x.shape[1]):
            out[:, d] = _moving_average(x[:, d], win)
        return out
    else:
        raise ValueError("Expected 1D or 2D array for moving average.")


def _fill_by_conf_or_nan(vals: np.ndarray, conf: Optional[np.ndarray], min_conf: float) -> np.ndarray:
    """
    Forward/back-fill low-confidence or NaN values, per-frame.
    - vals: [T, D], D=2 for (x,y)
    - conf: [T] or [T,1] (optional). If None, NaNs in vals are treated as 'bad' and filled.
    """
    vals = vals.astype(float, copy=True)
    T, D = vals.shape
    # Determine 'bad' mask per frame & dimension
    if conf is not None:
        c = conf.reshape(T, -1)[:, 0]
        bad = c < min_conf
        bad = np.repeat(bad[:, None], D, axis=1)  # broadcast to both x,y
        # If any NaNs present, treat them as bad too
        bad = bad | ~np.isfinite(vals)
    else:
        bad = ~np.isfinite(vals)

    # Forward fill (replace current bad with previous good)
    for t in range(1, T):
        vals[t][bad[t]] = vals[t-1][bad[t]]

    # Back fill for any leading bad run
    for t in range(T-2, -1, -1):
        vals[t][bad[t]] = vals[t+1][bad[t]]

    return vals


def _shoulder_angle_deg(Rs: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    """
    Angle (degrees) of vector from right-shoulder â†’ left-shoulder.
    Using image coords (x right, y down). arctan2(dy, dx).
    """
    dx = Ls[:, 0] - Rs[:, 0]
    dy = Ls[:, 1] - Rs[:, 1]
    return np.degrees(np.arctan2(dy, dx))


def _angle_between_points_deg(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Generalized angle (degrees) of vector from p1 â†’ p2.
    Using image coords (x right, y down). arctan2(dy, dx).
    p1, p2: [T, 2] arrays
    """
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    return np.degrees(np.arctan2(dy, dx))


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap degrees into [-180, 180)."""
    return (a + 180) % 360 - 180


def _seconds_axis(ax: plt.Axes, T: int, fps: float) -> None:
    """
    Add a top x-axis labeled in seconds aligned with the frame index on the bottom.
    """
    sec_ax = ax.secondary_xaxis('top', functions=(
        # frames -> seconds
        lambda frames: np.asarray(frames) / max(fps, 1e-9),
        # seconds -> frames
        lambda seconds: np.asarray(seconds) * max(fps, 1e-9)
    ))
    sec_ax.set_xlabel("time (s)")


# ------------------------------
# Core API - Generalized version
# ------------------------------
def plot_keypoints_and_angles(
    kps: np.ndarray,
    scores: Optional[np.ndarray] = None,
    params: UpperPlotParams = UpperPlotParams(),
    title: str = "Keypoint tracking over time",
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Create flexible plots for up to 10 keypoints and multiple angles.

    Supports two modes:
    1. Legacy mode (use_legacy_mode=True): Original wrist y-progress & shoulder angle behavior
    2. Flexible mode (use_legacy_mode=False): Custom keypoints and angles via params.keypoints and params.angles

    Parameters:
    -----------
    kps : np.ndarray [T, 17, 2]
        Keypoint coordinates
    scores : Optional[np.ndarray] [T, 17]
        Confidence scores for each keypoint
    params : PlotParams
        Configuration parameters
    title : str
        Plot title

    Returns:
    --------
    (fig, axes) : Tuple[plt.Figure, Tuple[plt.Axes, ...]]
    """

    # --------------------------
    # Basic validation
    # --------------------------
    if kps.ndim != 3 or kps.shape[1:] != (17, 2):
        raise ValueError("kps must have shape [T, 17, 2].")
    T = kps.shape[0]
    if scores is not None and (scores.ndim != 2 or scores.shape != (T, 17)):
        raise ValueError("scores must have shape [T, 17] if provided.")

    # --------------------------
    # Legacy mode: use original behavior
    # --------------------------
    if params.use_legacy_mode:
        return _plot_legacy_mode(kps, scores, params, title)

    # --------------------------
    # Flexible mode: use custom keypoints and angles
    # --------------------------
    if not params.keypoints and not params.angles:
        raise ValueError("In flexible mode, you must specify at least one keypoint or angle in params.")

    # Limit to 10 keypoints
    if len(params.keypoints) > 10:
        logger.warning(f"Too many keypoints specified ({len(params.keypoints)}). Using first 10.")
        params.keypoints = params.keypoints[:10]

    # Process all keypoints
    processed_kps = {}
    for kp_cfg in params.keypoints:
        if kp_cfg.index < 0 or kp_cfg.index >= 17:
            raise ValueError(f"Keypoint index {kp_cfg.index} out of range [0, 16]")

        kp_data = kps[:, kp_cfg.index, :].copy()
        kp_score = None if scores is None else scores[:, kp_cfg.index]

        # Fill and smooth
        kp_data = _fill_by_conf_or_nan(kp_data, kp_score, params.min_conf)
        kp_data = _moving_average(kp_data, params.smooth_win)

        processed_kps[kp_cfg.index] = (kp_data, kp_cfg)

    # Process all angles
    processed_angles = []
    for ang_cfg in params.angles:
        if ang_cfg.point1_idx < 0 or ang_cfg.point1_idx >= 17:
            raise ValueError(f"Angle point1_idx {ang_cfg.point1_idx} out of range [0, 16]")
        if ang_cfg.point2_idx < 0 or ang_cfg.point2_idx >= 17:
            raise ValueError(f"Angle point2_idx {ang_cfg.point2_idx} out of range [0, 16]")

        p1 = kps[:, ang_cfg.point1_idx, :].copy()
        p2 = kps[:, ang_cfg.point2_idx, :].copy()

        p1_score = None if scores is None else scores[:, ang_cfg.point1_idx]
        p2_score = None if scores is None else scores[:, ang_cfg.point2_idx]

        # Fill and smooth
        p1 = _fill_by_conf_or_nan(p1, p1_score, params.min_conf)
        p2 = _fill_by_conf_or_nan(p2, p2_score, params.min_conf)
        p1 = _moving_average(p1, params.smooth_win)
        p2 = _moving_average(p2, params.smooth_win)

        # Calculate angle
        angle_deg = _angle_between_points_deg(p1, p2)

        # Calculate delta if requested
        delta_deg = None
        if ang_cfg.show_delta:
            n = max(1, int(ang_cfg.delta_baseline_n))
            base_deg = np.nanmedian(angle_deg[:min(T, n)])
            delta_deg = _wrap_deg(angle_deg - base_deg)

        processed_angles.append((angle_deg, delta_deg, ang_cfg))

    # --------------------------
    # Determine subplot layout
    # --------------------------
    # Count how many panels we need
    n_coord_panels = 0
    has_y_coords = any(kp_cfg.plot_y for _, kp_cfg in processed_kps.values())
    has_x_coords = any(kp_cfg.plot_x for _, kp_cfg in processed_kps.values())
    if has_y_coords:
        n_coord_panels += 1
    if has_x_coords:
        n_coord_panels += 1

    n_angle_panels = len(processed_angles)
    n_panels = n_coord_panels + n_angle_panels

    if n_panels == 0:
        raise ValueError("No plots requested. Enable plot_y or plot_x for keypoints, or add angles.")

    # Create figure
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.4 * n_panels), sharex=False)
    if n_panels == 1:
        axes = (axes,)  # ensure tuple-like

    # --------------------------
    # Plot keypoint coordinates
    # --------------------------
    panel_idx = 0

    # Y-coordinates panel
    if has_y_coords:
        ax = axes[panel_idx]
        for kp_data, kp_cfg in processed_kps.values():
            if kp_cfg.plot_y:
                ax.plot(
                    kp_data[:, 1], 
                    label=f"{kp_cfg.label} y",
                    color=kp_cfg.color,
                    linestyle=kp_cfg.linestyle,
                    alpha=kp_cfg.alpha
                )
        ax.set_ylabel("y (px)")
        if params.invert_y:
            ax.invert_yaxis()
        ax.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.25)
        if params.fps and params.show_seconds_axis:
            _seconds_axis(ax, T, params.fps)
        panel_idx += 1

    # X-coordinates panel
    if has_x_coords:
        ax = axes[panel_idx]
        for kp_data, kp_cfg in processed_kps.values():
            if kp_cfg.plot_x:
                ax.plot(
                    kp_data[:, 0], 
                    label=f"{kp_cfg.label} x",
                    color=kp_cfg.color,
                    linestyle=kp_cfg.linestyle,
                    alpha=kp_cfg.alpha
                )
        ax.set_ylabel("x (px)")
        ax.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.25)
        if params.fps and params.show_seconds_axis:
            _seconds_axis(ax, T, params.fps)
        panel_idx += 1

    # --------------------------
    # Plot angles
    # --------------------------
    for angle_deg, delta_deg, ang_cfg in processed_angles:
        ax = axes[panel_idx]
        ax.plot(angle_deg, label=ang_cfg.name)
        if delta_deg is not None:
            ax.plot(delta_deg, linestyle="--", label=f"{ang_cfg.name} Î”", alpha=0.9)
            ax.axhline(+75, linestyle=":", alpha=0.35)
            ax.axhline(-75, linestyle=":", alpha=0.35)
        ax.set_ylabel("degrees")
        ax.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.25)
        if params.fps and params.show_seconds_axis:
            _seconds_axis(ax, T, params.fps)
        panel_idx += 1

    # Label bottom axis
    axes[-1].set_xlabel("frame")

    # Title
    wrapped = "\n".join(textwrap.wrap(title, width=80))
    fig.suptitle(wrapped)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    return fig, axes


def _plot_legacy_mode(
    kps: np.ndarray,
    scores: Optional[np.ndarray],
    params: UpperPlotParams,
    title: str
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Original plotting behavior: wrist y-progress & shoulder angle.
    This maintains backward compatibility.
    """
    T = kps.shape[0]
    idxs = _lead_trail_idxs(params.handedness)

    # Extract raw series
    Ls = kps[:, L_SH, :].copy()
    Rs = kps[:, R_SH, :].copy()
    lw = kps[:, idxs["lead_wr"], :].copy()
    rw = kps[:, idxs["trail_wr"], :].copy()

    # Fill low-confidence/NaNs
    Ls = _fill_by_conf_or_nan(Ls, None if scores is None else scores[:, L_SH], params.min_conf)
    Rs = _fill_by_conf_or_nan(Rs, None if scores is None else scores[:, R_SH], params.min_conf)
    lw = _fill_by_conf_or_nan(lw, None if scores is None else scores[:, idxs["lead_wr"]], params.min_conf)
    rw = _fill_by_conf_or_nan(rw, None if scores is None else scores[:, idxs["trail_wr"]], params.min_conf)

    # Smooth
    Ls_s = _moving_average(Ls, params.smooth_win)
    Rs_s = _moving_average(Rs, params.smooth_win)
    lw_s = _moving_average(lw, params.smooth_win)
    rw_s = _moving_average(rw, params.smooth_win)

    # Shoulder metrics
    shoulder_deg = _shoulder_angle_deg(Rs_s, Ls_s)

    # Î” from early-frame baseline
    shoulder_delta = None
    if params.show_shoulder_delta:
        n = max(1, int(params.shoulder_delta_baseline_n))
        base_deg = np.nanmedian(shoulder_deg[:min(T, n)])
        shoulder_delta = _wrap_deg(shoulder_deg - base_deg)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.4 * 2), sharex=False)

    # Panel 1: wrists y
    ax0 = axes[0]
    ax0.plot(lw_s[:, 1], label="Lead wrist y")
    ax0.plot(rw_s[:, 1], label="Trail wrist y", alpha=0.8)
    ax0.set_ylabel("y (px)")
    if params.invert_y:
        ax0.invert_yaxis()
    ax0.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax0.grid(True, alpha=0.25)
    if params.fps and params.show_seconds_axis:
        _seconds_axis(ax0, T, params.fps)

    # Panel 2: shoulder angles
    ax1 = axes[1]
    ax1.plot(shoulder_deg, label="Shoulder angle (deg)")
    if shoulder_delta is not None:
        ax1.plot(shoulder_delta, linestyle="--", label="Î” from early baseline (deg)", alpha=0.9)
        ax1.axhline(+75, linestyle=":", alpha=0.35)
        ax1.axhline(-75, linestyle=":", alpha=0.35)
    ax1.set_ylabel("degrees")
    ax1.set_xlabel("frame")
    ax1.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.25)

    # Title
    wrapped = "\n".join(textwrap.wrap(title, width=80))
    fig.suptitle(wrapped)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    return fig, axes


# Backward compatibility alias
def plot_wrist_y_and_shoulder_deg(
    kps: np.ndarray,
    scores: Optional[np.ndarray] = None,
    params: UpperPlotParams = UpperPlotParams(),
    title: str = "Wrists y-progress & Shoulder angle over time",
    include_normalized_hands: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Legacy function maintained for backward compatibility.
    Automatically uses legacy mode.
    """
    if include_normalized_hands:
        logger.warning("include_normalized_hands parameter is deprecated in the generalized version")

    # Force legacy mode
    params.use_legacy_mode = True
    return plot_keypoints_and_angles(kps, scores, params, title)

# === Auto-added on 2025-10-16T02:47:23.720182 ===

# ------------------------------
# Upper-body list comparison API
# ------------------------------
def _compute_upper_metrics(kps: np.ndarray,
                           scores: Optional[np.ndarray],
                           params: UpperPlotParams) -> Dict[str, np.ndarray]:
    """
    Compute time-series metrics for *one* sequence (upper body).
    Returns a dict of 1D arrays keyed by metric name:
      - 'wrist_y_lead'
      - 'wrist_y_trail'
      - 'shoulder_angle_deg'  (signed, absolute orientation of shoulder line vs horizontal)
      - 'shoulder_delta_deg'  (angle minus early-frame baseline; optional)
      - 'wrist_x_separation'  (absolute horizontal distance between wrists)
      - 'wrist_y_diff'        (lead_y - trail_y; sign reflects vertical spread)
    """
    # COCO indices we need
    L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10

    T = kps.shape[0]
    idxs = _lead_trail_idxs(params.handedness)

    # Extract raw series [T,2]
    shL = kps[:, idxs["lead_sh"], :].astype(float).copy()
    shT = kps[:, idxs["trail_sh"], :].astype(float).copy()
    wrL = kps[:, idxs["lead_wr"], :].astype(float).copy()
    wrT = kps[:, idxs["trail_wr"], :].astype(float).copy()

    # Confidence series per-joint, if provided
    c_shL = scores[:, idxs["lead_sh"]] if scores is not None else None
    c_shT = scores[:, idxs["trail_sh"]] if scores is not None else None
    c_wrL = scores[:, idxs["lead_wr"]] if scores is not None else None
    c_wrT = scores[:, idxs["trail_wr"]] if scores is not None else None

    # Fill low-confidence or NaNs (per (x,y), per joint)
    shL = _fill_by_conf_or_nan(shL, c_shL, params.min_conf)
    shT = _fill_by_conf_or_nan(shT, c_shT, params.min_conf)
    wrL = _fill_by_conf_or_nan(wrL, c_wrL, params.min_conf)
    wrT = _fill_by_conf_or_nan(wrT, c_wrT, params.min_conf)

    # Optionally smooth each coordinate independently
    if params.smooth_win and params.smooth_win > 1:
        shL[:, 0] = _moving_average(shL[:, 0], params.smooth_win)
        shL[:, 1] = _moving_average(shL[:, 1], params.smooth_win)
        shT[:, 0] = _moving_average(shT[:, 0], params.smooth_win)
        shT[:, 1] = _moving_average(shT[:, 1], params.smooth_win)
        wrL[:, 0] = _moving_average(wrL[:, 0], params.smooth_win)
        wrL[:, 1] = _moving_average(wrL[:, 1], params.smooth_win)
        wrT[:, 0] = _moving_average(wrT[:, 0], params.smooth_win)
        wrT[:, 1] = _moving_average(wrT[:, 1], params.smooth_win)

    # Prepare outputs (invert y if desired for "up is up" intuition)
    sgn = -1.0 if params.invert_y else 1.0
    wrist_y_lead  = sgn * wrL[:, 1]
    wrist_y_trail = sgn * wrT[:, 1]

    # Shoulder absolute angle vs horizontal (signed, in degrees)
    dxy = (shL - shT)  # vector from trail shoulder -> lead shoulder
    shoulder_angle_deg = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))  # [-180, 180)

    # Δ from early-frame baseline (median of first N frames)
    shoulder_delta_deg = None
    if getattr(params, "show_shoulder_delta", False):
        n = max(1, int(getattr(params, "shoulder_delta_baseline_n", 5)))
        base = np.median(shoulder_angle_deg[:n])
        shoulder_delta_deg = shoulder_angle_deg - base

    # Wrist separations
    wrist_x_separation = np.abs(wrL[:, 0] - wrT[:, 0])
    wrist_y_diff = (sgn * wrL[:, 1]) - (sgn * wrT[:, 1])

    out = dict(
        wrist_y_lead=wrist_y_lead,
        wrist_y_trail=wrist_y_trail,
        shoulder_angle_deg=shoulder_angle_deg,
        wrist_x_separation=wrist_x_separation,
        wrist_y_diff=wrist_y_diff
    )
    if shoulder_delta_deg is not None:
        out["shoulder_delta_deg"] = shoulder_delta_deg
    return out



def plot_upper_body_comparison(
    kps_list: List[np.ndarray],
    scores_list: Optional[List[np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    params: UpperPlotParams = UpperPlotParams(),
    title: str = "Swing Comparison - Upper Body",
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Compare *upper-body* kinematics across multiple swings/clips.
    This version splits shoulder metrics and wrist separations into their own panels
    to reduce visual clutter.

    Panels (6 total):
      0) Shoulder absolute angle (deg)
      1) Shoulder Δ from early baseline (deg)
      2) Lead wrist y
      3) Trail wrist y
      4) Wrist horizontal separation (x-sep)
      5) Wrist vertical difference (lead - trail) (y-diff)

    Parameters
    ----------
    kps_list : List[np.ndarray]
        Each element is [T, 17, 2] in COCO order.
    scores_list : Optional[List[np.ndarray]]
        Each element is [T, 17] confidence scores (optional; can pass None).
    labels : Optional[List[str]]
        Label per sequence; defaults to "Seq 1..N".
    params : UpperPlotParams
        Plot configuration (handedness, smoothing, fps, etc.).
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : Tuple[matplotlib.axes.Axes, ...]
        (ax0, ax1, ax2, ax3, ax4, ax5) as listed above.
    """
    logger = logging.getLogger("upper_body_comparison")
    if not isinstance(kps_list, (list, tuple)) or len(kps_list) == 0:
        raise ValueError("kps_list must be a non-empty list of [T,17,2] arrays.")
    N = len(kps_list)

    if scores_list is None:
        scores_list = [None] * N
    if labels is None or len(labels) != N:
        labels = [f"Seq {i+1}" for i in range(N)]

    # Validate shapes
    #for i, kps in enumerate(kps_list):
    #    if not isinstance(kps, np.ndarray) or kps.ndim != 3 or kps.shape[1:] != (17, 2):
    #        raise ValueError(f"kps_list[{i}] must have shape [T, 17, 2]")
    for i, sc in enumerate(scores_list):
        if sc is None:
            continue
        if not isinstance(sc, np.ndarray) or sc.ndim != 2 or sc.shape[1] != 17:
            raise ValueError(f"scores_list[{i}] must have shape [T, 17]")

    # Compute metrics per sequence
    metrics_list = []
    for kps, sc in zip(kps_list, scores_list):
        metrics = _compute_upper_metrics(kps, sc, params)
        # Ensure we always have a shoulder_delta_deg series for panel 1
        if 'shoulder_delta_deg' not in metrics:
            n = max(1, int(getattr(params, "shoulder_delta_baseline_n", 5)))
            base = np.median(metrics['shoulder_angle_deg'][:n])
            metrics['shoulder_delta_deg'] = metrics['shoulder_angle_deg'] - base
        metrics_list.append(metrics)

    # Set up figure: 6 panels to spread the content
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True, constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes


    # Color scheme aligned with lower_body.py
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    # import seaborn as sns

    # NUM_COLORS = 50
    # colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # linestyles = ['-', '--', '-.', ':'] * 13
    # linestyles = linestyles[:50]



    # Shoulder absolute angle
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax0.plot(metrics['shoulder_angle_deg'], label=f"{label}", linewidth=1.6, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax0.set_ylabel("shoulder°")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax0.set_title(title or "Swing Comparison - Upper Body", fontsize=16)

    # Optional seconds axis at the top
    if params.show_seconds_axis and params.fps:
        T_max = max(kps.shape[0] for kps in kps_list)
        _seconds_axis(ax0, T_max, params.fps)

    # Shoulder delta
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax1.plot(metrics['shoulder_delta_deg'], label=f"{label}", linewidth=1.6, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax1.set_ylabel("Δ shoulder°")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))

    # Lead wrist y
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax2.plot(metrics['wrist_y_lead'], label=f"{label}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax2.set_ylabel("lead wrist y" + (" (inv)" if params.invert_y else ""))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))

    # Trail wrist y
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax3.plot(metrics['wrist_y_trail'], label=f"{label}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax3.set_ylabel("trail wrist y" + (" (inv)" if params.invert_y else ""))
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))

    # Wrist x-separation
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax4.plot(metrics['wrist_x_separation'], label=f"{label}", linewidth=1.6, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax4.set_ylabel("wrist x-sep (px)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))

    # Wrist y-diff
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax5.plot(metrics['wrist_y_diff'], label=f"{label}", linewidth=1.6, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], alpha=0.85)
    ax5.set_ylabel("wrist y-diff (px)")
    ax5.set_xlabel("frame")
    ax5.grid(True, alpha=0.25)
    ax5.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))

    return fig, tuple(axes)
