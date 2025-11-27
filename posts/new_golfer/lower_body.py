import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import textwrap
from dataclasses import dataclass

# COCO keypoint indices (17-keypoint format)
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

@dataclass
class PlotParams:
    """Parameters for plotting"""
    handedness: str = "right"  # "left" or "right"
    min_conf: float = 0.3
    smooth_win: int = 5
    fps: Optional[float] = None
    show_seconds_axis: bool = True


def _fill_by_conf_or_nan(pts: np.ndarray, conf: Optional[np.ndarray], thresh: float) -> np.ndarray:
    """Fill low-confidence points with NaN, then forward/backward fill."""
    filled = pts.copy()
    if conf is not None:
        mask = conf < thresh
        filled[mask] = np.nan
    
    # Forward fill
    for i in range(1, len(filled)):
        if np.any(np.isnan(filled[i])):
            filled[i] = filled[i-1]
    # Backward fill
    for i in range(len(filled)-2, -1, -1):
        if np.any(np.isnan(filled[i])):
            filled[i] = filled[i+1]
    
    return filled


def _moving_average(pts: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average smoothing."""
    if window < 2:
        return pts
    
    smoothed = pts.copy()
    pad = window // 2
    for i in range(len(pts)):
        start = max(0, i - pad)
        end = min(len(pts), i + pad + 1)
        smoothed[i] = np.nanmean(pts[start:end], axis=0)
    
    return smoothed


def _angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Calculate angle at p2 formed by p1-p2-p3 (in degrees).
    
    Args:
        p1, p2, p3: Arrays of shape [T, 2] representing points over time
    
    Returns:
        Array of shape [T] with angles in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    dot = np.sum(v1 * v2, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_angle = dot / (mag1 * mag2 + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
    
    return angles


def _unwrap_angles(angles: np.ndarray, threshold: float = 180.0) -> np.ndarray:
    """
    Unwrap angle array to remove discontinuities.
    
    Similar to numpy.unwrap but for degrees.
    Adds/subtracts 360° to maintain continuity when jumps exceed threshold.
    """
    unwrapped = angles.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]
        if diff > threshold:
            unwrapped[i:] -= 360.0
        elif diff < -threshold:
            unwrapped[i:] += 360.0
    return unwrapped


def _hip_rotation_angle(r_hip: np.ndarray, l_hip: np.ndarray) -> np.ndarray:
    """Calculate hip rotation angle from horizontal."""
    delta = l_hip - r_hip
    angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    # Add unwrapping to prevent discontinuities
    return _unwrap_angles(angles)



def _lead_trail_idxs(handedness: str) -> dict:
    """Get lead/trail indices based on handedness."""
    if handedness.lower() == "right":
        return {
            "lead_hip": L_HIP, "trail_hip": R_HIP,
            "lead_knee": L_KNEE, "trail_knee": R_KNEE,
            "lead_ankle": L_ANKLE, "trail_ankle": R_ANKLE,
        }
    else:
        return {
            "lead_hip": R_HIP, "trail_hip": L_HIP,
            "lead_knee": R_KNEE, "trail_knee": L_KNEE,
            "lead_ankle": R_ANKLE, "trail_ankle": L_ANKLE,
        }


def _seconds_axis(ax, num_frames: int, fps: float):
    """Add secondary x-axis showing seconds."""
    ax2 = ax.twiny()
    ax2.set_xlim(0, num_frames / fps)
    ax2.set_xlabel("time (s)")


def _compute_lower_body_metrics(
    kps: np.ndarray,
    scores: Optional[np.ndarray],
    params: PlotParams
) -> dict:
    """
    Compute all lower body metrics for a single keypoint array.
    
    Returns dict with keys:
        - hip_rotation_deg
        - lead_hip_lateral_shift
        - lead_knee_flexion
        - trail_knee_flexion
        - knee_hip_ratio
        - lead_ankle_sway
        - trail_ankle_sway
    """
    T = kps.shape[0]
    idxs = _lead_trail_idxs(params.handedness)
    
    # Extract raw series
    l_hip = kps[:, L_HIP, :].copy()
    r_hip = kps[:, R_HIP, :].copy()
    l_knee = kps[:, L_KNEE, :].copy()
    r_knee = kps[:, R_KNEE, :].copy()
    l_ankle = kps[:, L_ANKLE, :].copy()
    r_ankle = kps[:, R_ANKLE, :].copy()
    
    # Fill low-confidence/NaNs
    l_hip = _fill_by_conf_or_nan(l_hip, None if scores is None else scores[:, L_HIP], params.min_conf)
    r_hip = _fill_by_conf_or_nan(r_hip, None if scores is None else scores[:, R_HIP], params.min_conf)
    l_knee = _fill_by_conf_or_nan(l_knee, None if scores is None else scores[:, L_KNEE], params.min_conf)
    r_knee = _fill_by_conf_or_nan(r_knee, None if scores is None else scores[:, R_KNEE], params.min_conf)
    l_ankle = _fill_by_conf_or_nan(l_ankle, None if scores is None else scores[:, L_ANKLE], params.min_conf)
    r_ankle = _fill_by_conf_or_nan(r_ankle, None if scores is None else scores[:, R_ANKLE], params.min_conf)
    
    # Smooth with larger window for lower body
    smooth_win_lb = max(params.smooth_win * 2, 7)
    
    l_hip_s = _moving_average(l_hip, smooth_win_lb)
    r_hip_s = _moving_average(r_hip, smooth_win_lb)
    l_knee_s = _moving_average(l_knee, smooth_win_lb)
    r_knee_s = _moving_average(r_knee, smooth_win_lb)
    l_ankle_s = _moving_average(l_ankle, smooth_win_lb)
    r_ankle_s = _moving_average(r_ankle, smooth_win_lb)
    
    # Calculate metrics
    hip_rotation_deg = _hip_rotation_angle(r_hip_s, l_hip_s)
    
    lead_hip_pts = l_hip_s if params.handedness == "right" else r_hip_s
    address_x = np.nanmedian(lead_hip_pts[:min(10, T), 0])
    lead_hip_lateral_shift = lead_hip_pts[:, 0] - address_x
    
    lead_knee_flexion = _angle_between_points(
        l_hip_s if params.handedness == "right" else r_hip_s,
        l_knee_s if params.handedness == "right" else r_knee_s,
        l_ankle_s if params.handedness == "right" else r_ankle_s
    )
    trail_knee_flexion = _angle_between_points(
        r_hip_s if params.handedness == "right" else l_hip_s,
        r_knee_s if params.handedness == "right" else l_knee_s,
        r_ankle_s if params.handedness == "right" else l_ankle_s
    )
    
    knee_width = np.linalg.norm(l_knee_s - r_knee_s, axis=1)
    hip_width = np.linalg.norm(l_hip_s - r_hip_s, axis=1)
    knee_hip_ratio = knee_width / (hip_width + 1e-6)
    
    lead_ankle_pts = l_ankle_s if params.handedness == "right" else r_ankle_s
    trail_ankle_pts = r_ankle_s if params.handedness == "right" else l_ankle_s
    ankle_address_lead = np.nanmedian(lead_ankle_pts[:min(10, T), 0])
    ankle_address_trail = np.nanmedian(trail_ankle_pts[:min(10, T), 0])
    lead_ankle_sway = lead_ankle_pts[:, 0] - ankle_address_lead
    trail_ankle_sway = trail_ankle_pts[:, 0] - ankle_address_trail
    
    return {
        'hip_rotation_deg': hip_rotation_deg,
        'lead_hip_lateral_shift': lead_hip_lateral_shift,
        'lead_knee_flexion': lead_knee_flexion,
        'trail_knee_flexion': trail_knee_flexion,
        'knee_hip_ratio': knee_hip_ratio,
        'lead_ankle_sway': lead_ankle_sway,
        'trail_ankle_sway': trail_ankle_sway,
    }


def plot_lower_body_comparison(
    kps_list: List[np.ndarray],
    scores_list: Optional[List[np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    params: PlotParams = PlotParams(),
    title: str = "Lower Body Kinematics Comparison",
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Compare lower body kinematics across multiple swings/clips.
    
    Args:
        kps_list: List of keypoint arrays, each of shape [T, 17, 2]
        scores_list: Optional list of score arrays, each of shape [T, 17]
        labels: Labels for each swing (e.g., ["Swing 1", "Swing 2", ...])
        params: Plotting parameters
        title: Figure title
    
    Returns:
        (fig, axes) tuple
    """
    
    # Validation
    n_swings = len(kps_list)
    # if n_swings > 4:
    #     raise ValueError("Maximum 4 swings supported for comparison")
    
    if scores_list is None:
        scores_list = [None] * n_swings
    
    if labels is None:
        labels = [f"Swing {i+1}" for i in range(n_swings)]
    
    # Validate all inputs
    for i, kps in enumerate(kps_list):
        if kps.ndim != 3 or kps.shape[1:] != (17, 2):
            raise ValueError(f"kps_list[{i}] must have shape [T, 17, 2]")
        if scores_list[i] is not None:
            T = kps.shape[0]
            if scores_list[i].shape != (T, 17):
                raise ValueError(f"scores_list[{i}] must have shape [{T}, 17]")
    
    # Compute metrics for all swings
    metrics_list = []
    for kps, scores in zip(kps_list, scores_list):
        metrics = _compute_lower_body_metrics(kps, scores, params)
        metrics_list.append(metrics)
    
    # Color scheme
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    # linestyles = ['-', '--', '-.', ':']
    # New (supports 10)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # import seaborn as sns

    # NUM_COLORS = 50
    # colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # linestyles = ['-', '--', '-.', ':'] * 13
    # linestyles = linestyles[:50]


    
    # Create figure
    #fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    # Dynamic figure sizing based on number of swings
    fig_height = 14 + (n_swings - 4) if n_swings > 4 else 14
    fig, axes = plt.subplots(4, 1, figsize=(12, fig_height), sharex=False)

    
    # -------------
    # Panel 1: Hip rotation angles
    # -------------
    ax0 = axes[0]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax0.plot(metrics['hip_rotation_deg'], 
                label=f"{label}", 
                color=colors[i], 
                linestyle=linestyles[i],
                alpha=0.85)
    
    ax0.set_ylabel("Hip rotation (deg)")
    #ax0.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax0.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Hip Rotation Angle Comparison")
    
    if params.fps and params.show_seconds_axis:
        T_max = max(kps.shape[0] for kps in kps_list)
        _seconds_axis(ax0, T_max, params.fps)
    
    # -------------
    # Panel 2: Lead hip lateral shift
    # -------------
    ax1 = axes[1]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax1.plot(metrics['lead_hip_lateral_shift'], 
                label=f"{label}", 
                color=colors[i], 
                linestyle=linestyles[i],
                alpha=0.85)
    
    ax1.axhline(0, linestyle=':', alpha=0.3, color='gray', label='Address position')
    ax1.set_ylabel("Lateral shift (px)")
    ax1.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Lead Hip Lateral Weight Transfer")
    
    # -------------
    # Panel 3: Knee flexion (lead knee focus)
    # -------------
    ax2 = axes[2]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax2.plot(metrics['lead_knee_flexion'], 
                label=f"{label} - Lead knee", 
                color=colors[i], 
                linestyle=linestyles[i],
                alpha=0.85)
    
    ax2.axhline(180, linestyle=':', alpha=0.3, color='gray', label='Fully extended')
    ax2.axhline(160, linestyle=':', alpha=0.3, color='gray')
    ax2.set_ylabel("Flexion angle (deg)")
    ax2.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Lead Knee Flexion Comparison")
    #ax2.set_ylim([140, 190])
    
    # -------------
    # Panel 4: Knee/Hip ratio
    # -------------
    ax3 = axes[3]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax3.plot(metrics['knee_hip_ratio'], 
                label=f"{label}", 
                color=colors[i], 
                linestyle=linestyles[i],
                alpha=0.85)
    
    ax3.axhline(1.0, linestyle=':', alpha=0.3, color='gray', label='Equal width')
    ax3.set_ylabel("Ratio")
    ax3.set_xlabel("frame")
    ax3.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    ax3.grid(True, alpha=0.25)
    ax3.set_title("Knee Separation Ratio (knee width / hip width)")
    
    # Title + layout
    wrapped = "\n".join(textwrap.wrap(title, width=80))
    fig.suptitle(wrapped, fontsize=6, y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    return fig, axes


def plot_lower_body_detailed(
    kps_list: List[np.ndarray],
    scores_list: Optional[List[np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    params: PlotParams = PlotParams(),
    title: str = "Detailed Lower Body Analysis",
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """
    Create detailed 4x4 grid showing all metrics for up to 4 swings.
    Each row = one metric, each column overlays all swings for that metric.
    
    Args:
        kps_list: List of keypoint arrays (up to 4), each of shape [T, 17, 2]
        scores_list: Optional list of score arrays
        labels: Labels for each swing
        params: Plotting parameters
        title: Figure title
    
    Returns:
        (fig, axes) tuple where axes is shape (7, 1)
    """
    
    # Validation
    n_swings = len(kps_list)
    #if n_swings > 4:
    #    raise ValueError("Maximum 4 swings supported")
    
    if scores_list is None:
        scores_list = [None] * n_swings
    
    if labels is None:
        labels = [f"Swing {i+1}" for i in range(n_swings)]
    
    # Compute metrics
    metrics_list = []
    for kps, scores in zip(kps_list, scores_list):
        metrics = _compute_lower_body_metrics(kps, scores, params)
        metrics_list.append(metrics)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    # import seaborn as sns

    # NUM_COLORS = 50
    # colors = sns.color_palette('husl', n_colors=NUM_COLORS)
    # linestyles = ['-', '--', '-.', ':'] * 13
    # linestyles = linestyles[:50]
    
    # Create comprehensive figure with 7 panels
    fig, axes = plt.subplots(7, 1, figsize=(12, 20), sharex=False)
    
    # 1. Hip rotation
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[0].plot(metrics['hip_rotation_deg'], label=label, 
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[0].set_ylabel("Degrees")
    axes[0].set_title("Hip Rotation Angle")
    axes[0].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[0].grid(True, alpha=0.25)
    
    # 2. Lead hip lateral shift
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[1].plot(metrics['lead_hip_lateral_shift'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[1].axhline(0, linestyle=':', alpha=0.3, color='gray')
    axes[1].set_ylabel("Pixels")
    axes[1].set_title("Lead Hip Lateral Shift")
    axes[1].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[1].grid(True, alpha=0.25)
    
    # 3. Lead knee flexion
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[2].plot(metrics['lead_knee_flexion'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[2].axhline(180, linestyle=':', alpha=0.3, color='gray')
    axes[2].set_ylabel("Degrees")
    axes[2].set_title("Lead Knee Flexion")
    axes[2].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[2].grid(True, alpha=0.25)
    #axes[2].set_ylim([140, 190])
    
    # 4. Trail knee flexion
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[3].plot(metrics['trail_knee_flexion'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[3].axhline(180, linestyle=':', alpha=0.3, color='gray')
    axes[3].set_ylabel("Degrees")
    axes[3].set_title("Trail Knee Flexion")
    axes[3].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[3].grid(True, alpha=0.25)
    #axes[3].set_ylim([140, 190])
    
    # 5. Knee/hip ratio
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[4].plot(metrics['knee_hip_ratio'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[4].axhline(1.0, linestyle=':', alpha=0.3, color='gray')
    axes[4].set_ylabel("Ratio")
    axes[4].set_title("Knee/Hip Width Ratio")
    axes[4].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[4].grid(True, alpha=0.25)
    
    # 6. Lead ankle sway
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[5].plot(metrics['lead_ankle_sway'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[5].axhline(0, linestyle=':', alpha=0.3, color='gray')
    axes[5].set_ylabel("Pixels")
    axes[5].set_title("Lead Ankle Lateral Sway")
    axes[5].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[5].grid(True, alpha=0.25)
    
    # 7. Trail ankle sway
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        axes[6].plot(metrics['trail_ankle_sway'], label=label,
                    color=colors[i], linestyle=linestyles[i], alpha=0.85)
    axes[6].axhline(0, linestyle=':', alpha=0.3, color='gray')
    axes[6].set_ylabel("Pixels")
    axes[6].set_xlabel("Frame")
    axes[6].set_title("Trail Ankle Lateral Sway")
    axes[6].legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
    axes[6].grid(True, alpha=0.25)
    
    wrapped = "\n".join(textwrap.wrap(title, width=80))
    fig.suptitle(wrapped, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    return fig, axes





# # COCO keypoint indices (17-keypoint format)
# L_HIP = 11
# R_HIP = 12
# L_KNEE = 13
# R_KNEE = 14
# L_ANKLE = 15
# R_ANKLE = 16

# @dataclass
# class HipPlotParams:
#     """Parameters for plotting (matching your existing structure)"""
#     handedness: str = "right"  # "left" or "right"
#     min_conf: float = 0.3
#     smooth_win: int = 5  # Use larger window for lower body (e.g., 7-9)
#     fps: Optional[float] = None
#     show_seconds_axis: bool = True
#     invert_y: bool = True


# def _fill_by_conf_or_nan(pts: np.ndarray, conf: Optional[np.ndarray], thresh: float) -> np.ndarray:
#     """Fill low-confidence points with NaN, then forward/backward fill."""
#     filled = pts.copy()
#     if conf is not None:
#         mask = conf < thresh
#         filled[mask] = np.nan
    
#     # Forward fill
#     for i in range(1, len(filled)):
#         if np.any(np.isnan(filled[i])):
#             filled[i] = filled[i-1]
#     # Backward fill
#     for i in range(len(filled)-2, -1, -1):
#         if np.any(np.isnan(filled[i])):
#             filled[i] = filled[i+1]
    
#     return filled


# def _moving_average(pts: np.ndarray, window: int) -> np.ndarray:
#     """Centered moving average smoothing."""
#     if window < 2:
#         return pts
    
#     smoothed = pts.copy()
#     pad = window // 2
#     for i in range(len(pts)):
#         start = max(0, i - pad)
#         end = min(len(pts), i + pad + 1)
#         smoothed[i] = np.nanmean(pts[start:end], axis=0)
    
#     return smoothed


# def _angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
#     """
#     Calculate angle at p2 formed by p1-p2-p3 (in degrees).
    
#     Args:
#         p1, p2, p3: Arrays of shape [T, 2] representing points over time
    
#     Returns:
#         Array of shape [T] with angles in degrees (0-180)
#     """
#     # Vectors from p2 to p1 and p2 to p3
#     v1 = p1 - p2  # [T, 2]
#     v2 = p3 - p2  # [T, 2]
    
#     # Magnitudes
#     mag1 = np.linalg.norm(v1, axis=1)  # [T]
#     mag2 = np.linalg.norm(v2, axis=1)  # [T]
    
#     # Dot product
#     dot = np.sum(v1 * v2, axis=1)  # [T]
    
#     # Angle in radians, then degrees
#     with np.errstate(divide='ignore', invalid='ignore'):
#         cos_angle = dot / (mag1 * mag2 + 1e-8)
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         angles = np.degrees(np.arccos(cos_angle))
    
#     return angles


# def _hip_rotation_angle(r_hip: np.ndarray, l_hip: np.ndarray) -> np.ndarray:
#     """
#     Calculate hip rotation angle (similar to shoulder angle).
#     Positive = right hip forward, negative = left hip forward.
    
#     Returns angle in degrees from horizontal.
#     """
#     delta = l_hip - r_hip  # [T, 2]
#     angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
#     return angles


# def _lead_trail_idxs(handedness: str) -> dict:
#     """Get lead/trail indices based on handedness."""
#     if handedness.lower() == "right":
#         return {
#             "lead_hip": L_HIP, "trail_hip": R_HIP,
#             "lead_knee": L_KNEE, "trail_knee": R_KNEE,
#             "lead_ankle": L_ANKLE, "trail_ankle": R_ANKLE,
#         }
#     else:
#         return {
#             "lead_hip": R_HIP, "trail_hip": L_HIP,
#             "lead_knee": R_KNEE, "trail_knee": L_KNEE,
#             "lead_ankle": R_ANKLE, "trail_ankle": L_ANKLE,
#         }


# def _seconds_axis(ax, num_frames: int, fps: float):
#     """Add secondary x-axis showing seconds."""
#     ax2 = ax.twiny()
#     ax2.set_xlim(0, num_frames / fps)
#     ax2.set_xlabel("time (s)")


# def plot_lower_body_kinematics(
#     kps: np.ndarray,
#     scores: Optional[np.ndarray] = None,
#     params: HipPlotParams = HipPlotParams(),
#     title: str = "Lower Body Kinematics: Hip Rotation, Knee Flexion & Stability",
# ) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
#     """
#     Create plots for lower body golf swing analysis:
#       - Panel 1: Hip rotation angle + lateral displacement of lead hip
#       - Panel 2: Knee flexion angles for both legs
#       - Panel 3: Knee separation ratio (knee width / hip width)
#       - Panel 4: Ankle lateral positions (to detect swaying)
    
#     Args:
#         kps: Keypoints array of shape [T, 17, 2]
#         scores: Optional confidence scores of shape [T, 17]
#         params: Plotting parameters
#         title: Figure title
    
#     Returns:
#         (fig, axes) tuple
#     """
    
#     # --------------------------
#     # Basic validation
#     # --------------------------
#     if kps.ndim != 3 or kps.shape[1:] != (17, 2):
#         raise ValueError("kps must have shape [T, 17, 2].")
#     T = kps.shape[0]
#     if scores is not None and (scores.ndim != 2 or scores.shape != (T, 17)):
#         raise ValueError("scores must have shape [T, 17] if provided.")
    
#     # Identify lead/trail indexes
#     idxs = _lead_trail_idxs(params.handedness)
    
#     # --------------------------
#     # Extract raw series
#     # --------------------------
#     l_hip = kps[:, L_HIP, :].copy()
#     r_hip = kps[:, R_HIP, :].copy()
#     l_knee = kps[:, L_KNEE, :].copy()
#     r_knee = kps[:, R_KNEE, :].copy()
#     l_ankle = kps[:, L_ANKLE, :].copy()
#     r_ankle = kps[:, R_ANKLE, :].copy()
    
#     # --------------------------
#     # Fill low-confidence/NaNs
#     # --------------------------
#     l_hip = _fill_by_conf_or_nan(l_hip, None if scores is None else scores[:, L_HIP], params.min_conf)
#     r_hip = _fill_by_conf_or_nan(r_hip, None if scores is None else scores[:, R_HIP], params.min_conf)
#     l_knee = _fill_by_conf_or_nan(l_knee, None if scores is None else scores[:, L_KNEE], params.min_conf)
#     r_knee = _fill_by_conf_or_nan(r_knee, None if scores is None else scores[:, R_KNEE], params.min_conf)
#     l_ankle = _fill_by_conf_or_nan(l_ankle, None if scores is None else scores[:, L_ANKLE], params.min_conf)
#     r_ankle = _fill_by_conf_or_nan(r_ankle, None if scores is None else scores[:, R_ANKLE], params.min_conf)
    
#     # --------------------------
#     # Smooth (use larger window for lower body - typically 2-3x upper body)
#     # --------------------------
#     smooth_win_lb = max(params.smooth_win * 2, 7)  # Larger window for noisier lower body
    
#     l_hip_s = _moving_average(l_hip, smooth_win_lb)
#     r_hip_s = _moving_average(r_hip, smooth_win_lb)
#     l_knee_s = _moving_average(l_knee, smooth_win_lb)
#     r_knee_s = _moving_average(r_knee, smooth_win_lb)
#     l_ankle_s = _moving_average(l_ankle, smooth_win_lb)
#     r_ankle_s = _moving_average(r_ankle, smooth_win_lb)
    
#     # --------------------------
#     # Calculate metrics
#     # --------------------------
    
#     # 1. Hip rotation angle (analogous to shoulder angle)
#     hip_rotation_deg = _hip_rotation_angle(r_hip_s, l_hip_s)
    
#     # 2. Lateral displacement of lead hip (x-position change from address)
#     lead_hip_pts = l_hip_s if params.handedness == "right" else r_hip_s
#     address_x = np.nanmedian(lead_hip_pts[:min(10, T), 0])  # baseline from first 10 frames
#     lead_hip_lateral_shift = lead_hip_pts[:, 0] - address_x  # pixels
    
#     # 3. Knee flexion angles (hip-knee-ankle triplets)
#     lead_knee_flexion = _angle_between_points(
#         l_hip_s if params.handedness == "right" else r_hip_s,
#         l_knee_s if params.handedness == "right" else r_knee_s,
#         l_ankle_s if params.handedness == "right" else r_ankle_s
#     )
#     trail_knee_flexion = _angle_between_points(
#         r_hip_s if params.handedness == "right" else l_hip_s,
#         r_knee_s if params.handedness == "right" else l_knee_s,
#         r_ankle_s if params.handedness == "right" else l_ankle_s
#     )
    
#     # 4. Knee separation ratio (normalized)
#     knee_width = np.linalg.norm(l_knee_s - r_knee_s, axis=1)
#     hip_width = np.linalg.norm(l_hip_s - r_hip_s, axis=1)
#     knee_hip_ratio = knee_width / (hip_width + 1e-6)
    
#     # 5. Ankle positions for sway detection
#     lead_ankle_pts = l_ankle_s if params.handedness == "right" else r_ankle_s
#     trail_ankle_pts = r_ankle_s if params.handedness == "right" else l_ankle_s
#     ankle_address_lead = np.nanmedian(lead_ankle_pts[:min(10, T), 0])
#     ankle_address_trail = np.nanmedian(trail_ankle_pts[:min(10, T), 0])
#     lead_ankle_sway = lead_ankle_pts[:, 0] - ankle_address_lead
#     trail_ankle_sway = trail_ankle_pts[:, 0] - ankle_address_trail
    
#     # ---------------------------------
#     # Prepare figure / axes
#     # ---------------------------------
#     fig, axes = plt.subplots(4, 1, figsize=(10, 13.6), sharex=False)
    
#     # -------------
#     # Panel 1: Hip rotation + lateral displacement
#     # -------------
#     ax0 = axes[0]
#     ax0_twin = ax0.twinx()
    
#     l1 = ax0.plot(hip_rotation_deg, label="Hip rotation angle (deg)", color='tab:blue')
#     l2 = ax0_twin.plot(lead_hip_lateral_shift, label="Lead hip lateral shift (px)", 
#                         color='tab:orange', linestyle='--', alpha=0.8)
    
#     ax0.set_ylabel("Hip rotation (deg)", color='tab:blue')
#     ax0_twin.set_ylabel("Lateral shift (px)", color='tab:orange')
#     ax0.tick_params(axis='y', labelcolor='tab:blue')
#     ax0_twin.tick_params(axis='y', labelcolor='tab:orange')
    
#     # Combine legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
#     lns = l1 + l2
#     labs = [l.get_label() for l in lns]
#     ax0.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
#     ax0.grid(True, alpha=0.25)
#     ax0.set_title("Hip Rotation & Weight Transfer")
    
#     if params.fps and params.show_seconds_axis:
#         _seconds_axis(ax0, T, params.fps)
    
#     # -------------
#     # Panel 2: Knee flexion angles
#     # -------------
#     ax1 = axes[1]
#     ax1.plot(lead_knee_flexion, label="Lead knee flexion (deg)", color='tab:green')
#     ax1.plot(trail_knee_flexion, label="Trail knee flexion (deg)", 
#              color='tab:red', alpha=0.8, linestyle='--')
    
#     # Reference lines for common flexion ranges
#     ax1.axhline(180, linestyle=':', alpha=0.3, color='gray', label='Fully extended')
#     ax1.axhline(160, linestyle=':', alpha=0.3, color='gray')
    
#     ax1.set_ylabel("Flexion angle (deg)")
#     ax1.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
#     ax1.grid(True, alpha=0.25)
#     ax1.set_title("Knee Flexion Dynamics")
#     ax1.set_ylim([140, 190])  # Typical golf range
    
#     # -------------
#     # Panel 3: Knee/Hip separation ratio
#     # -------------
#     ax2 = axes[2]
#     ax2.plot(knee_hip_ratio, label="Knee width / Hip width", color='tab:purple')
#     ax2.axhline(1.0, linestyle=':', alpha=0.5, color='gray', label='Equal width')
    
#     ax2.set_ylabel("Ratio")
#     ax2.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
#     ax2.grid(True, alpha=0.25)
#     ax2.set_title("Knee Separation (indicates hip rotation quality)")
    
#     # -------------
#     # Panel 4: Ankle sway detection
#     # -------------
#     ax3 = axes[3]
#     ax3.plot(lead_ankle_sway, label="Lead ankle lateral sway (px)", color='tab:cyan')
#     ax3.plot(trail_ankle_sway, label="Trail ankle lateral sway (px)", 
#              color='tab:brown', alpha=0.8, linestyle='--')
    
#     ax3.axhline(0, linestyle=':', alpha=0.5, color='gray', label='Address position')
#     ax3.set_ylabel("Lateral sway (px)")
#     ax3.set_xlabel("frame")
#     ax3.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.02, 1))
#     ax3.grid(True, alpha=0.25)
#     ax3.set_title("Ankle Stability (excessive sway = loss of axis)")
    
#     # -------------
#     # Title + layout polish
#     # -------------
#     wrapped = "\n".join(textwrap.wrap(title, width=80))
#     fig.suptitle(wrapped, fontsize=14, y=0.995)
#     fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    
#     return fig, axes


# # Example usage:
# # fig, axes = plot_lower_body_kinematics(
# #     kps=your_keypoints,  # [T, 17, 2]
# #     scores=your_scores,  # [T, 17]
# #     params=PlotParams(handedness="right", smooth_win=5, fps=30.0)
# # )
# # plt.show()
