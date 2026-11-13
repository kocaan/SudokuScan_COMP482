from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Order a four-point set as TL, TR, BR, BL for roughly upright puzzles."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)
    if len(np.unique(np.round(ordered, 2), axis=0)) != 4:
        # Angle ordering is a safe fallback for symmetric/rotated configurations.
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        cyclic = pts[np.argsort(angles)]
        start = int(np.argmin(cyclic.sum(axis=1)))
        cyclic = np.roll(cyclic, -start, axis=0)
        # Ensure the second point is on the right for a near-upright puzzle.
        if cyclic[1, 0] < cyclic[-1, 0]:
            cyclic = cyclic[[0, 3, 2, 1]]
        ordered = cyclic.astype(np.float32)
    return ordered


def target_corners(size: int) -> np.ndarray:
    # The assignment explicitly defines the ideal unused A3 corner as (450, 450).
    return np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)


def affine_least_squares(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit a 2x3 affine transform to all supplied point pairs by least squares."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    design = np.zeros((2 * len(src), 6), dtype=np.float64)
    values = dst.reshape(-1)
    for i, (x, y) in enumerate(src):
        design[2 * i] = [x, y, 1, 0, 0, 0]
        design[2 * i + 1] = [0, 0, 0, x, y, 1]
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    return solution.reshape(2, 3).astype(np.float64)


def build_rectifications(corners: np.ndarray, size: int) -> dict[str, dict[str, np.ndarray | float]]:
    """Create assignment models A3, A4 and H4 and their residual diagnostics."""
    src = order_points(corners).astype(np.float32)
    dst = target_corners(size)

    # A3: TL, TR, BL determine the affine map; BR is deliberately unused.
    a3 = cv2.getAffineTransform(src[[0, 1, 3]], dst[[0, 1, 3]])
    mapped_br_a3 = cv2.transform(src[2:3].reshape(1, 1, 2), a3)[0, 0]
    a3_error = float(np.linalg.norm(mapped_br_a3 - dst[2]))

    # OpenCV's estimateAffine2D exposes RANSAC and LMEDS rather than a plain
    # least-squares method. Give RANSAC a deliberately generous reprojection
    # threshold so all four correspondences are inliers; its refinement stage
    # then performs the all-four-point least-squares fit required for A4.
    # (Using the default threshold would incorrectly reject a corner because a
    # perspective quadrilateral is not exactly affine.)
    a4, a4_inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=1_000_000.0,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if a4 is None or a4_inliers is None or int(a4_inliers.sum()) != 4:
        # Numerical fallback: the explicit normal-equation least-squares fit
        # uses every correspondence, never a three-point subset.
        a4 = affine_least_squares(src, dst)
        a4_inlier_count = 4
        a4_estimator_fallback = True
    else:
        a4_inlier_count = int(a4_inliers.sum())
        a4_estimator_fallback = False
    a4 = np.asarray(a4, dtype=np.float64)
    mapped_a4 = cv2.transform(src.reshape(1, -1, 2), a4)[0]
    a4_rms = float(np.sqrt(np.mean(np.sum((mapped_a4 - dst) ** 2, axis=1))))

    h4 = cv2.getPerspectiveTransform(src, dst)
    mapped_h4 = cv2.perspectiveTransform(src.reshape(1, -1, 2), h4)[0]
    h4_rms = float(np.sqrt(np.mean(np.sum((mapped_h4 - dst) ** 2, axis=1))))

    return {
        "A3": {"matrix": a3.astype(np.float64), "corner_error_px": a3_error},
        "A4": {"matrix": a4.astype(np.float64), "rms_px": a4_rms,
               "estimator_inliers": a4_inlier_count,
               "estimator_fallback": a4_estimator_fallback},
        "H4": {"matrix": h4.astype(np.float64), "rms_px": h4_rms},
    }


def obliqueness_measure(corners: np.ndarray) -> float:
    """Perspective-side imbalance: max opposite-side ratio minus one."""
    p = order_points(corners)
    lengths = np.array([
        np.linalg.norm(p[1] - p[0]),
        np.linalg.norm(p[2] - p[3]),
        np.linalg.norm(p[3] - p[0]),
        np.linalg.norm(p[2] - p[1]),
    ])
    ratios = [lengths[0] / max(lengths[1], 1e-6), lengths[1] / max(lengths[0], 1e-6),
              lengths[2] / max(lengths[3], 1e-6), lengths[3] / max(lengths[2], 1e-6)]
    return float(max(ratios) - 1.0)


def grid_line_deviations(rectified: np.ndarray, size: int = 450) -> dict[str, list[float] | float]:
    """Locate 10 vertical/horizontal grid lines and measure offsets and spacing."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    inv = 255.0 - gray.astype(np.float32)
    x_profile = inv.mean(axis=0)
    y_profile = inv.mean(axis=1)
    radius = max(3, int(round(size / 25)))
    expected = np.linspace(0, size, 10)

    def locate(profile: np.ndarray) -> tuple[list[float], list[float]]:
        positions: list[float] = []
        deviations: list[float] = []
        for value in expected:
            center = int(round(value))
            lo, hi = max(0, center - radius), min(size, center + radius + 1)
            if lo >= hi:
                found = float(value)
            else:
                local = profile[lo:hi]
                found = float(lo + int(np.argmax(local)))
            positions.append(found)
            deviations.append(float(found - value))
        return positions, deviations

    vertical_positions, dx = locate(x_profile)
    horizontal_positions, dy = locate(y_profile)
    combined = np.asarray(dx + dy, dtype=np.float64)
    ideal_spacing = size / 9.0
    vertical_spacing = np.diff(np.asarray(vertical_positions, dtype=np.float64))
    horizontal_spacing = np.diff(np.asarray(horizontal_positions, dtype=np.float64))
    spacing_errors = np.concatenate([vertical_spacing - ideal_spacing,
                                     horizontal_spacing - ideal_spacing])
    return {
        "vertical_px": dx,
        "horizontal_px": dy,
        "vertical_positions_px": vertical_positions,
        "horizontal_positions_px": horizontal_positions,
        "vertical_spacing_px": vertical_spacing.tolist(),
        "horizontal_spacing_px": horizontal_spacing.tolist(),
        "rms_px": float(np.sqrt(np.mean(combined ** 2))),
        "max_abs_px": float(np.max(np.abs(combined))),
        "spacing_rms_px": float(np.sqrt(np.mean(spacing_errors ** 2))),
        "spacing_max_abs_px": float(np.max(np.abs(spacing_errors))),
    }

