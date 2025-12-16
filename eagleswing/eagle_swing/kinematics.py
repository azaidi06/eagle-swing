import numpy as np



def angle_2points_deg(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    # Compute arctan2
    angles = np.arctan2(dy, dx)
    # Convert to degrees in-place to save memory allocation
    np.degrees(angles, out=angles)
    return angles


def angle_3points_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Returns signed angle (degrees) at vertex b, from vector ba -> bc.
    Range: [-180, 180]
    Positive = Clockwise (if y-axis is down/image coords)
    """
    # Create vectors relative to B
    ba = a - b
    bc = c - b
    
    # Calculate determinant (2D cross product) and dot product
    # det = x1*y2 - y1*x2
    det = ba[:, 0] * bc[:, 1] - ba[:, 1] * bc[:, 0]
    
    # dot = x1*x2 + y1*y2
    dot = ba[:, 0] * bc[:, 0] + ba[:, 1] * bc[:, 1]
    
    # arctan2(y, x) -> arctan2(det, dot)
    angles = np.arctan2(det, dot)
    
    # In-place conversion to degrees
    np.degrees(angles, out=angles)
    return angles