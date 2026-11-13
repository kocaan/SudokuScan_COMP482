from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict

import cv2
import numpy as np

from .config import ScanConfig
from .geometry import order_points


@dataclass
class CornerDetection:
    corners: np.ndarray  # TL, TR, BR, BL in detection-image coordinates
    candidates: np.ndarray  # Harris candidate coordinates, N x 2
    response: np.ndarray
    score: float
    proposal: np.ndarray
    diagnostics: dict


def _extract_harris_candidates(gray: np.ndarray, cfg: ScanConfig) -> tuple[np.ndarray, np.ndarray]:
    source = np.float32(gray)
    response = cv2.cornerHarris(source, cfg.harris_block_size, cfg.harris_ksize, cfg.harris_k)
    response = cv2.GaussianBlur(response, (3, 3), 0)
    max_response = float(np.max(response))
    if not np.isfinite(max_response) or max_response <= 0:
        return np.empty((0, 2), np.float32), response

    threshold = cfg.response_threshold * max_response
    nms_radius = max(1, int(round(min(gray.shape) * cfg.nms_radius_fraction)))
    kernel_size = 2 * nms_radius + 1
    dilated = cv2.dilate(response, np.ones((kernel_size, kernel_size), np.uint8))
    mask = (response >= threshold) & (response >= dilated - max(1e-12, abs(max_response) * 1e-7))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32), response

    values = response[ys, xs]
    order = np.argsort(values)[::-1]
    minimum_distance = max(2.0, min(gray.shape) * cfg.min_candidate_distance_fraction)
    minimum_distance_sq = minimum_distance * minimum_distance
    selected: list[tuple[float, float]] = []
    # Spatial buckets avoid the quadratic Python scan over all previously
    # accepted candidates while preserving the same Euclidean NMS rule.
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for idx in order:
        point = (float(xs[idx]), float(ys[idx]))
        bucket = (int(point[0] // minimum_distance), int(point[1] // minimum_distance))
        too_close = False
        for bx in range(bucket[0] - 1, bucket[0] + 2):
            for by in range(bucket[1] - 1, bucket[1] + 2):
                for prior in buckets.get((bx, by), ()):
                    dx, dy = point[0] - prior[0], point[1] - prior[1]
                    if dx * dx + dy * dy < minimum_distance_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if not too_close:
            selected.append(point)
            buckets[bucket].append(point)
            if len(selected) >= cfg.max_candidates:
                break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2), response



def _extract_harris_candidates_multiscale(
    gray: np.ndarray, cfg: ScanConfig
) -> tuple[np.ndarray, np.ndarray]:

    base_candidates, base_response = _extract_harris_candidates(gray, cfg)
    response_maps = [base_response]
    candidate_sets = [base_candidates]
    supplemental = [(3, 3, 0.06), (5, 3, 0.04)]
    for block_size, ksize, harris_k in supplemental:
        if (block_size, ksize, harris_k) == (
            cfg.harris_block_size, cfg.harris_ksize, cfg.harris_k
        ):
            continue
        variant = replace(
            cfg, harris_block_size=block_size, harris_ksize=ksize,
            harris_k=harris_k,
        )
        candidates, response = _extract_harris_candidates(gray, variant)
        if len(candidates):
            candidate_sets.append(candidates)
            response_maps.append(response)

    if len(candidate_sets) == 1:
        return base_candidates, base_response

    normalized_maps: list[np.ndarray] = []
    for response in response_maps:
        positive = np.maximum(response, 0).astype(np.float32)
        maximum = float(positive.max())
        normalized_maps.append(positive / maximum if maximum > 0 else positive)
    combined_response = np.maximum.reduce(normalized_maps).astype(np.float32)

    all_candidates = np.concatenate(candidate_sets, axis=0)
    ys = np.clip(np.rint(all_candidates[:, 1]).astype(int), 0, gray.shape[0] - 1)
    xs = np.clip(np.rint(all_candidates[:, 0]).astype(int), 0, gray.shape[1] - 1)
    values = combined_response[ys, xs]
    order = np.argsort(values)[::-1]
    minimum_distance = max(2.0, min(gray.shape) * cfg.min_candidate_distance_fraction)
    minimum_distance_sq = minimum_distance * minimum_distance
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    selected: list[tuple[float, float]] = []
    for index in order:
        point = (float(all_candidates[index, 0]), float(all_candidates[index, 1]))
        bucket = (int(point[0] // minimum_distance), int(point[1] // minimum_distance))
        too_close = False
        for bx in range(bucket[0] - 1, bucket[0] + 2):
            for by in range(bucket[1] - 1, bucket[1] + 2):
                for prior in buckets.get((bx, by), ()):
                    dx, dy = point[0] - prior[0], point[1] - prior[1]
                    if dx * dx + dy * dy < minimum_distance_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if not too_close:
            selected.append(point)
            buckets[bucket].append(point)
            if len(selected) >= cfg.max_candidates:
                break

    return np.asarray(selected, dtype=np.float32).reshape(-1, 2), combined_response

def _contour_quads(gray: np.ndarray, binary: np.ndarray, cfg: ScanConfig) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    image_area = float(h * w)
    proposal_images: list[np.ndarray] = [binary]
    edges = cv2.Canny(gray, 45, 130)
    close_size = max(3, int(round(min(h, w) * 0.003)))
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    proposal_images.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2))

    found: list[tuple[float, np.ndarray]] = []
    for proposal_image in proposal_images:
        contours, _ = cv2.findContours(proposal_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:250]
        for contour in contours:
            area = float(cv2.contourArea(contour))
            fraction = area / image_area
            if not (cfg.proposal_min_area_fraction <= fraction <= cfg.proposal_max_area_fraction):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
            else:

                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect).astype(np.float32)
            quad = order_points(quad)
            found.append((area, quad))

    result: list[np.ndarray] = []
    for _area, quad in sorted(found, key=lambda z: z[0], reverse=True):
        center = quad.mean(axis=0)
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        duplicate = False
        for prior in result:
            prior_center = prior.mean(axis=0)
            prior_area = abs(cv2.contourArea(prior.astype(np.float32)))
            if np.linalg.norm(center - prior_center) < 0.025 * min(h, w) and min(area, prior_area) / max(area, prior_area, 1.0) > 0.78:
                duplicate = True
                break
        if not duplicate:
            result.append(quad)
        if len(result) >= cfg.max_proposals:
            break
    return result



def _hough_grid_quads(gray: np.ndarray, cfg: ScanConfig) -> list[np.ndarray]:

    h, w = gray.shape[:2]
    minimum_side = float(min(h, w))
    edges = cv2.Canny(gray, 40, 130)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(28, int(round(minimum_side * 0.035))),
        minLineLength=max(36, int(round(minimum_side * 0.10))),
        maxLineGap=max(8, int(round(minimum_side * 0.03))),
    )
    if lines is None:
        return []


    try:
        line_array = np.asarray(lines)
        if line_array.size == 0 or line_array.size % 4 != 0:
            return []
        line_segments = line_array.reshape(-1, 4).astype(np.float64, copy=False)
    except (TypeError, ValueError):

        return []

    segments: list[tuple[float, float, float, float, float, float]] = []
    for x1, y1, x2, y2 in line_segments:
        if not np.all(np.isfinite((x1, y1, x2, y2))):
            continue
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 0.09 * minimum_side:
            continue
        angle = float(np.arctan2(dy, dx) % np.pi)
        segments.append((x1, y1, x2, y2, length, angle))
    if len(segments) < 4:
        return []
    segments.sort(key=lambda item: item[4], reverse=True)
    segments = segments[:180]

    angles = np.asarray([item[5] for item in segments], dtype=np.float64)
    weights = np.asarray([item[4] for item in segments], dtype=np.float64)
    vectors = np.column_stack((np.cos(2.0 * angles), np.sin(2.0 * angles)))
    pair_distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(pair_distances)), pair_distances.shape)
    if pair_distances[first, second] < 0.85:  # less than about 25 degrees apart
        return []
    centers = np.asarray([vectors[first], vectors[second]], dtype=np.float64)
    labels = np.full(len(segments), -1, dtype=np.int32)
    for _ in range(20):
        distances = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        updated = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for family in range(2):
            selected = labels == family
            if not np.any(selected):
                return []
            mean = np.sum(vectors[selected] * weights[selected, None], axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 1e-9:
                return []
            centers[family] = mean / norm
    if min(int(np.count_nonzero(labels == 0)), int(np.count_nonzero(labels == 1))) < 2:
        return []

    image_center = np.asarray([w * 0.5, h * 0.5], dtype=np.float64)
    tolerance = max(3.0, 0.006 * minimum_side)
    family_groups: list[list[dict[str, object]]] = []
    family_axes: list[tuple[np.ndarray, np.ndarray]] = []
    for family in range(2):
        member_indices = np.flatnonzero(labels == family)
        mean_angle = 0.5 * float(np.arctan2(centers[family, 1], centers[family, 0]))
        direction = np.asarray([np.cos(mean_angle), np.sin(mean_angle)], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        family_axes.append((direction, normal))
        located: list[tuple[float, float, int]] = []
        for index in member_indices:
            x1, y1, x2, y2, length, _angle = segments[int(index)]
            a, b = y1 - y2, x2 - x1
            c = x1 * y2 - x2 * y1
            denominator = a * normal[0] + b * normal[1]
            if abs(denominator) < 1e-6:
                continue
            offset = -(a * image_center[0] + b * image_center[1] + c) / denominator
            located.append((float(offset), float(length), int(index)))
        located.sort(key=lambda item: item[0])

        groups: list[dict[str, object]] = []
        for offset, length, index in located:
            if not groups or offset - float(groups[-1]["offset"]) > tolerance:
                groups.append({"items": [(offset, length, index)], "offset": offset, "weight": length})
            else:
                group = groups[-1]
                items = group["items"]
                assert isinstance(items, list)
                items.append((offset, length, index))
                group["offset"] = float(np.average([item[0] for item in items], weights=[item[1] for item in items]))
                group["weight"] = float(group["weight"]) + length

        groups = [group for group in groups if float(group["weight"]) >= 0.07 * minimum_side]
        if len(groups) < 2:
            return []

        chosen_ids: list[int] = []
        chosen_ids.extend(range(min(5, len(groups))))
        chosen_ids.extend(range(max(0, len(groups) - 5), len(groups)))
        strongest = sorted(range(len(groups)), key=lambda idx: float(groups[idx]["weight"]), reverse=True)[:4]
        chosen_ids.extend(strongest)
        chosen_ids = sorted(set(chosen_ids))
        family_groups.append([groups[idx] for idx in chosen_ids])

    def group_line(group: dict[str, object]) -> np.ndarray | None:
        items = group["items"]
        assert isinstance(items, list)
        coeffs: list[np.ndarray] = []
        coeff_weights: list[float] = []
        for _offset, length, index in items:
            x1, y1, x2, y2, _seglen, _angle = segments[index]
            a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
            norm = float(np.hypot(a, b))
            if norm <= 1e-9:
                continue
            coeffs.append(np.asarray([a, b, c], dtype=np.float64) / norm)
            coeff_weights.append(length)
        if not coeffs:
            return None
        reference = coeffs[0][:2]
        adjusted = [coeff if np.dot(coeff[:2], reference) >= 0 else -coeff for coeff in coeffs]
        mean = np.average(np.stack(adjusted), axis=0, weights=coeff_weights)
        norm = float(np.hypot(mean[0], mean[1]))
        if norm <= 1e-9:
            return None
        return mean / norm

    def intersect(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
        matrix = np.asarray([[line_a[0], line_a[1]], [line_b[0], line_b[1]]], dtype=np.float64)
        if abs(float(np.linalg.det(matrix))) < 1e-5:
            return None
        return np.linalg.solve(matrix, -np.asarray([line_a[2], line_b[2]], dtype=np.float64))

    proposals: list[tuple[float, np.ndarray]] = []
    for pair_a in __import__("itertools").combinations(family_groups[0], 2):
        if abs(float(pair_a[0]["offset"]) - float(pair_a[1]["offset"])) < 0.13 * minimum_side:
            continue
        line_a1, line_a2 = group_line(pair_a[0]), group_line(pair_a[1])
        if line_a1 is None or line_a2 is None:
            continue
        for pair_b in __import__("itertools").combinations(family_groups[1], 2):
            if abs(float(pair_b[0]["offset"]) - float(pair_b[1]["offset"])) < 0.13 * minimum_side:
                continue
            line_b1, line_b2 = group_line(pair_b[0]), group_line(pair_b[1])
            if line_b1 is None or line_b2 is None:
                continue
            intersections = [
                intersect(line_a1, line_b1), intersect(line_a1, line_b2),
                intersect(line_a2, line_b2), intersect(line_a2, line_b1),
            ]
            if any(point is None for point in intersections):
                continue
            quad = order_points(np.asarray(intersections, dtype=np.float32))
            if np.any(quad[:, 0] < -0.08 * w) or np.any(quad[:, 0] > 1.08 * w):
                continue
            if np.any(quad[:, 1] < -0.08 * h) or np.any(quad[:, 1] > 1.08 * h):
                continue
            area = abs(float(cv2.contourArea(quad)))
            if area < 0.008 * h * w:
                continue
            if not _quad_is_plausible(quad, (h, w)):
                continue
            proposals.append((area, quad))

    unique: list[tuple[float, np.ndarray]] = []
    for area, quad in sorted(proposals, key=lambda item: item[0], reverse=True):
        center = quad.mean(axis=0)
        duplicate = False
        for prior_area, prior in unique:
            center_distance = float(np.linalg.norm(center - prior.mean(axis=0)))
            area_ratio = min(area, prior_area) / max(area, prior_area, 1.0)
            corner_distance = float(np.mean(np.linalg.norm(quad - prior, axis=1)))
            if center_distance < 0.018 * minimum_side and area_ratio > 0.88 and corner_distance < 0.035 * minimum_side:
                duplicate = True
                break
        if not duplicate:
            unique.append((area, quad))
        if len(unique) >= cfg.max_hough_proposals:
            break
    return [quad for _area, quad in unique]

def _quad_is_plausible(quad: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    p = order_points(quad)
    lengths = np.array([np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)])
    if not cv2.isContourConvex(p):
        return False
    if np.min(lengths) < 0.035 * min(h, w):
        return False
    area = abs(cv2.contourArea(p))
    if area < 0.008 * h * w or area > 0.93 * h * w:
        return False

    opposing = max(lengths[0], lengths[2]) / max(min(lengths[0], lengths[2]), 1.0)
    vertical = max(lengths[1], lengths[3]) / max(min(lengths[1], lengths[3]), 1.0)
    return opposing < 7.0 and vertical < 7.0


def _candidate_match(proposal: np.ndarray, candidates: np.ndarray, response: np.ndarray,
                     cfg: ScanConfig) -> tuple[np.ndarray | None, float, list[float]]:
    h, w = response.shape[:2]
    radius = max(12.0, min(h, w) * cfg.corner_match_radius_fraction)
    max_response = max(float(response.max()), 1e-9)
    chosen: list[np.ndarray] = []
    distances: list[float] = []
    response_scores: list[float] = []
    used_candidates: set[int] = set()

    for vertex in order_points(proposal):
        if len(candidates) == 0:
            return None, -1.0, []
        delta = candidates - vertex[None, :]
        dists = np.linalg.norm(delta, axis=1)
        nearby = np.where(dists <= radius)[0]
        if len(nearby) == 0:
            return None, -1.0, distances

        vals = response[candidates[nearby, 1].astype(int), candidates[nearby, 0].astype(int)] / max_response
        #scores = 0.78 * (1.0 - dists[nearby] / radius) + 0.22 * np.clip(vals, 0.0, 1.0)
        scores = ( 0.90 * (1.0 - dists[nearby] / radius) + 0.10 * np.clip(vals, 0.0, 1.0))

        for used_index in used_candidates:
            local = np.flatnonzero(nearby == used_index)
            if local.size:
                scores[int(local[0])] = -np.inf
        best_local = int(np.argmax(scores))
        if not np.isfinite(scores[best_local]):
            return None, -1.0, distances
        best = int(nearby[best_local])
        used_candidates.add(best)
        chosen.append(candidates[best])
        distances.append(float(dists[best]))
        response_scores.append(float(response[candidates[best, 1].astype(int), candidates[best, 0].astype(int)] / max_response))



    corners = order_points(np.asarray(chosen, np.float32))
    if len(np.unique(np.round(corners, 1), axis=0)) != 4:
        return None, -1.0, distances
    mean_distance_score = 1.0 - float(np.mean(distances)) / radius
    mean_harris_score = float(np.mean(response_scores))
    return corners, 0.55 * mean_harris_score + 0.45 * mean_distance_score, distances


"""
def _proposal_grid_evidence(gray: np.ndarray, proposal: np.ndarray,
                            size: int = 450) -> tuple[np.ndarray, float]:
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    H = cv2.getPerspectiveTransform(order_points(proposal).astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    inv = 255.0 - warped.astype(np.float32)
    x_profile = inv.mean(axis=0)
    y_profile = inv.mean(axis=1)
    radius = 3
    line_strengths = []
    for profile in (x_profile, y_profile):
        for loc in np.linspace(0, size - 1, 10):
            center = int(round(loc))
            lo, hi = max(0, center - radius), min(size, center + radius + 1)
            local_peak = float(np.max(profile[lo:hi]))
            background = float(np.median(profile))
            line_strengths.append(max(0.0, local_peak - background) / 255.0)
    line_score = float(np.mean(line_strengths)) if line_strengths else 0.0
    return H, line_score
"""



def _proposal_grid_evidence(
    gray: np.ndarray,
    proposal: np.ndarray,
    size: int = 450,
) -> tuple[np.ndarray, float]:

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        order_points(proposal).astype(np.float32), dst
    )

    warped = cv2.warpPerspective(
        gray, H, (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    block = max(21, int(round(size * 0.07)))
    if block % 2 == 0:
        block += 1

    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(warped, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        7,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    long_length = max(15, int(round(size * 0.55)))

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (long_length, 1)
        ),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, long_length)
        ),
    )

    def family_score(
        line_mask: np.ndarray,
        axis: int,
    ) -> float:
        profile = np.mean(line_mask > 0, axis=axis)

        expected = np.linspace(0, size - 1, 10)
        radius = max(4, int(round(size * 0.02)))

        observed = []
        strengths = []

        for location in expected:
            center = int(round(location))
            lo = max(0, center - radius)
            hi = min(size, center + radius + 1)

            local = profile[lo:hi]
            peak_index = int(np.argmax(local)) + lo
            peak_strength = float(np.max(local))

            observed.append(float(peak_index))
            strengths.append(float(np.clip(
                peak_strength / 0.65, 0.0, 1.0
            )))

        coverage = (
            0.70 * float(np.mean(strengths))
            + 0.30 * float(np.percentile(strengths, 25))
        )

        ideal_spacing = (size - 1) / 9.0
        errors = np.diff(observed) - ideal_spacing

        spacing_error = float(
            np.sqrt(np.mean(errors ** 2))
        )

        spacing_score = float(np.exp(
            -spacing_error / max(1.0, size * 0.02)
        ))

        return 0.80 * coverage + 0.20 * spacing_score

    horizontal_score = family_score(horizontal, axis=1)
    vertical_score = family_score(vertical, axis=0)

    line_score = float(np.sqrt(
        horizontal_score * vertical_score
    ))

    return H, line_score



def _grid_evidence(gray: np.ndarray, proposal: np.ndarray, candidates: np.ndarray,
                   cached: tuple[np.ndarray, float] | None = None) -> tuple[float, dict]:
    size = 450
    H, line_score = cached if cached is not None else _proposal_grid_evidence(gray, proposal, size)

    candidate_score = 0.0
    node_hits = 0
    if len(candidates):
        warped_candidates = cv2.perspectiveTransform(candidates.reshape(1, -1, 2), H)[0]
        spacing = size / 9.0
        nearest_x = np.rint(warped_candidates[:, 0] / spacing).astype(np.int32)
        nearest_y = np.rint(warped_candidates[:, 1] / spacing).astype(np.int32)
        valid = ((nearest_x >= 0) & (nearest_x <= 9) &
                 (nearest_y >= 0) & (nearest_y <= 9))
        dx = warped_candidates[:, 0] - nearest_x * spacing
        dy = warped_candidates[:, 1] - nearest_y * spacing
        close = valid & ((dx * dx + dy * dy) < (7.5 * 7.5))
        node_ids = nearest_y[close] * 10 + nearest_x[close]
        node_hits = int(np.unique(node_ids).size)
        candidate_score = node_hits / 100.0
    combined = 0.52 * line_score + 0.48 * candidate_score
    return combined, {"grid_line_evidence": line_score,
                      "grid_intersection_coverage": candidate_score,
                      "grid_intersection_hits": node_hits}


def _refine(
    gray: np.ndarray,
    corners: np.ndarray,
    proposal: np.ndarray | None = None,
) -> np.ndarray:

    points = order_points(np.asarray(corners, dtype=np.float32)).reshape(-1, 1, 2).copy()
    anchors = order_points(proposal) if proposal is not None else points.reshape(-1, 2).copy()
    side_lengths = np.array([
        np.linalg.norm(anchors[(i + 1) % 4] - anchors[i]) for i in range(4)
    ], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    for index in range(4):
        point = points[index:index + 1]
        adjacent_scale = float((side_lengths[index - 1] + side_lengths[index]) * 0.5)
        displacement = float(np.linalg.norm(point[0, 0] - anchors[index]))
        if displacement > max(4.0, 0.018 * adjacent_scale):
            half_window = int(np.clip(round(0.065 * adjacent_scale), 7, 29))
        else:
            half_window = 5
        try:
            cv2.cornerSubPix(gray, point, (half_window, half_window), (-1, -1), criteria)
        except cv2.error:
            continue
    return order_points(points.reshape(-1, 2))


def detect_outer_corners(
    gray: np.ndarray,
    binary: np.ndarray,
    cfg: ScanConfig | None = None,
    *,
    precomputed_candidates_response: tuple[np.ndarray, np.ndarray] | None = None,
    precomputed_proposals: list[np.ndarray] | None = None,
    precomputed_grid_evidence: list[tuple[np.ndarray, float]] | None = None,
) -> CornerDetection | None:

    cfg = cfg or ScanConfig()
    if precomputed_candidates_response is None:
        candidates, response = _extract_harris_candidates_multiscale(gray, cfg)
    else:
        candidates, response = precomputed_candidates_response
    if len(candidates) < 4:
        return None

    proposals = (list(precomputed_proposals) if precomputed_proposals is not None
                 else _contour_quads(gray, binary, cfg))
    if precomputed_proposals is None:
        hough_proposals = _hough_grid_quads(gray, cfg)
        proposals.extend(hough_proposals)
        deduplicated: list[np.ndarray] = []
        for quad in proposals:
            center = quad.mean(axis=0)
            area = abs(float(cv2.contourArea(quad)))
            duplicate = False
            for prior in deduplicated:
                prior_center = prior.mean(axis=0)
                prior_area = abs(float(cv2.contourArea(prior)))
                if (np.linalg.norm(center - prior_center) < 0.012 * min(gray.shape)
                        and min(area, prior_area) / max(area, prior_area, 1.0) > 0.90
                        and float(np.mean(np.linalg.norm(quad - prior, axis=1))) < 0.02 * min(gray.shape)):
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append(quad)
        proposals = deduplicated[: cfg.max_proposals + cfg.max_hough_proposals]

    if not proposals:
        ys, xs = np.where(binary > 0)
        if len(xs) > 100:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            hull = cv2.convexHull(pts)
            rect = cv2.minAreaRect(hull)
            proposals.append(order_points(cv2.boxPoints(rect)))

    best: CornerDetection | None = None
    h, w = gray.shape[:2]
    for proposal_index, proposal in enumerate(proposals):
        if not _quad_is_plausible(proposal, (h, w)):
            continue
        matched, match_score, distances = _candidate_match(proposal, candidates, response, cfg)
        if matched is None:
            continue
        if not _quad_is_plausible(matched, (h, w)):
            continue

        cache = None
        if (precomputed_grid_evidence is not None
                and proposal_index < len(precomputed_grid_evidence)
                and max(distances, default=0.0) <= 2.0):
            cache = precomputed_grid_evidence[proposal_index]
        grid_score, grid_diag = _grid_evidence(gray, matched, candidates, cache)
        if grid_score < 0.10:
            continue
        area_fraction = abs(cv2.contourArea(order_points(matched))) / float(h * w)

        total_score = 5.0 * grid_score + 0.55 * match_score + 0.10 * np.sqrt(area_fraction)
        diagnostics = {
            **grid_diag,
            "candidate_match_score": float(match_score),
            "corner_match_distances_px": distances,
            "proposal_area_fraction": float(area_fraction),
            "proposal_count": len(proposals),
            "candidate_count": len(candidates),
            "grid_evidence_based_on": "matched_harris_quadrilateral",
        }
        if best is None or total_score > best.score:
            best = CornerDetection(
                corners=_refine(gray, matched, proposal),
                candidates=candidates,
                response=response,
                score=float(total_score),
                proposal=order_points(proposal),
                diagnostics=diagnostics,
            )
    if best is not None:

        if (best.diagnostics.get("grid_line_evidence", 0.0) >= 0.07
                and best.diagnostics.get("grid_intersection_coverage", 0.0) >= 0.06):
            return best


    ys, xs = np.where(binary > 0)
    if len(xs) > 100:
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        proposal = order_points(cv2.boxPoints(rect))
        if _quad_is_plausible(proposal, (h, w)):
            matched, score, distances = _candidate_match(proposal, candidates, response, cfg)
            if matched is not None and _quad_is_plausible(matched, (h, w)):
                return CornerDetection(
                    corners=_refine(gray, matched, proposal), candidates=candidates, response=response,
                    score=float(score), proposal=proposal,
                    diagnostics={"fallback": True, "corner_match_distances_px": distances,
                                 "candidate_count": len(candidates)},
                )
    return None


def render_harris_debug(gray: np.ndarray, response: np.ndarray, candidates: np.ndarray,
                        selected: np.ndarray | None = None) -> np.ndarray:
    """Visualize the Harris response map, all NMS candidates and selected corners."""
    positive = np.maximum(response, 0)
    if float(positive.max()) > 0:
        heat = cv2.normalize(positive, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        heat = np.zeros_like(gray, dtype=np.uint8)
    vis = cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR)
    for x, y in candidates:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    if selected is not None:
        for idx, (x, y) in enumerate(order_points(selected)):
            cv2.circle(vis, (int(round(x)), int(round(y))), 7, (0, 255, 0), 2)
            cv2.putText(vis, str(idx), (int(x) + 7, int(y) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Harris candidates: {len(candidates)}", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return vis
