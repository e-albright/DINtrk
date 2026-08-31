from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QRect, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSizePolicy, QWidget, QLabel, QPushButton, QFileDialog, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QProgressBar, QMessageBox, QListWidget, QListWidgetItem
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cv_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def percentile_clip(arr: np.ndarray, pct: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    pct = float(np.clip(pct, 0.0, 100.0))
    if pct <= 0:
        return arr
    vmax = np.percentile(arr, pct)
    if vmax <= 0:
        return arr
    return np.clip(arr, 0, vmax)


def apply_gamma(img01: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(1e-6, float(gamma))
    return np.power(np.clip(img01, 0.0, 1.0), gamma)


def colormap_heatmap(hm01: np.ndarray) -> np.ndarray:
    hm8 = (np.clip(hm01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(hm8, cv2.COLORMAP_TURBO)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(np.array(pvals, dtype=np.float64))
    adj = np.zeros(m, dtype=np.float64)
    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        val = float(pvals[idx]) * m / float(rank)
        prev = min(prev, val)
        adj[idx] = prev
    return np.clip(adj, 0.0, 1.0).tolist()


def underexplored_analysis_grid(
    centers: Dict[int, List[Tuple[float, float]]],
    arena_w: int,
    arena_h: int,
    grid_cols: int,
    grid_rows: int,
    alpha: float,
    mc_iterations: int = 4000,
    random_seed: int = 2026,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    grid_cols = int(max(2, grid_cols))
    grid_rows = int(max(2, grid_rows))
    alpha = float(np.clip(alpha, 1e-6, 0.5))
    mc_iterations = int(max(200, mc_iterations))
    rng = np.random.default_rng(int(random_seed))

    x_edges = np.array([int(round(i * arena_w / grid_cols)) for i in range(grid_cols + 1)], dtype=np.int32)
    y_edges = np.array([int(round(i * arena_h / grid_rows)) for i in range(grid_rows + 1)], dtype=np.int32)

    counts = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    all_pts: List[Tuple[float, float]] = []
    for tid_pts in centers.values():
        all_pts.extend(tid_pts)

    for x, y in all_pts:
        gx = int(np.clip((float(x) / max(float(arena_w), 1.0)) * grid_cols, 0, grid_cols - 1))
        gy = int(np.clip((float(y) / max(float(arena_h), 1.0)) * grid_rows, 0, grid_rows - 1))
        counts[gy, gx] += 1

    m = int(grid_rows * grid_cols)
    total = int(counts.sum())
    counts_flat = counts.reshape(-1)

    cell_areas = np.zeros(m, dtype=np.float64)
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            x0 = int(x_edges[gx])
            x1 = int(x_edges[gx + 1])
            y0 = int(y_edges[gy])
            y1 = int(y_edges[gy + 1])
            idx = gy * grid_cols + gx
            cell_areas[idx] = float(max(1, x1 - x0) * max(1, y1 - y0))
    probs = cell_areas / float(np.sum(cell_areas))
    expected = float(total) * probs
    sd = np.sqrt(np.maximum(float(total) * probs * (1.0 - probs), 1e-12))

    if total > 0:
        leq = np.zeros(m, dtype=np.int64)
        remain = mc_iterations
        batch = 250
        while remain > 0:
            b = min(batch, remain)
            sims = rng.multinomial(total, probs, size=b)
            leq += np.sum(sims <= counts_flat[None, :], axis=0)
            remain -= b
        pvals_np = (leq.astype(np.float64) + 1.0) / float(mc_iterations + 1)
    else:
        pvals_np = np.ones(m, dtype=np.float64)

    rows: List[Dict[str, float]] = []
    z_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            idx = gy * grid_cols + gx
            obs = int(counts_flat[idx])
            exp = float(expected[idx])
            z = float((obs - exp) / float(sd[idx])) if total > 0 else 0.0
            p = float(pvals_np[idx])
            z_grid[gy, gx] = float(z)

            x0 = int(x_edges[gx])
            x1 = int(x_edges[gx + 1])
            y0 = int(y_edges[gy])
            y1 = int(y_edges[gy + 1])
            rows.append({
                "cell_row": int(gy),
                "cell_col": int(gx),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(max(x0 + 1, x1)),
                "y1": int(max(y0 + 1, y1)),
                "observed": int(obs),
                "expected": float(exp),
                "null_prob": float(probs[idx]),
                "z_score": float(z),
                "p_value_low_tail": float(p),
                "p_value_fdr": 1.0,
                "is_underexplored": 0
            })

    padj = benjamini_hochberg(pvals_np.tolist())
    for i, row in enumerate(rows):
        row["p_value_fdr"] = float(padj[i])
        row["is_underexplored"] = int(
            (row["observed"] < row["expected"]) and (row["p_value_fdr"] < alpha)
        )

    return rows, z_grid


def gaussian_splat(heat: np.ndarray, x: float, y: float, sigma: float, weight: float = 1.0) -> None:
    h, w = heat.shape[:2]
    sigma = float(max(0.1, sigma))
    rad = int(max(2, math.ceil(3.0 * sigma)))
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - rad)
    x1 = min(w - 1, cx + rad)
    y0 = max(0, cy - rad)
    y1 = min(h - 1, cy + rad)
    if x0 > x1 or y0 > y1:
        return
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    gx = np.exp(-0.5 * ((xs - x) / sigma) ** 2)
    gy = np.exp(-0.5 * ((ys - y) / sigma) ** 2)
    blob = (gy[:, None] * gx[None, :]) * float(weight)
    heat[y0:y1 + 1, x0:x1 + 1] += blob.astype(np.float32)


def clamp_xywh(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = int(np.clip(x, 0, max(0, W - 2)))
    y = int(np.clip(y, 0, max(0, H - 2)))
    w = int(np.clip(w, 2, max(2, W - x)))
    h = int(np.clip(h, 2, max(2, H - y)))
    return x, y, w, h


def contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2.0, y + h / 2.0)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def component_containing_point(bin01: np.ndarray, px: int, py: int) -> Optional[np.ndarray]:
    if bin01.dtype != np.uint8:
        bin01 = bin01.astype(np.uint8)
    H, W = bin01.shape[:2]
    if not (0 <= px < W and 0 <= py < H):
        return None
    cc = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    labels = cc[1]
    lab = labels[py, px]
    if lab == 0:
        return None
    return (labels == lab).astype(np.uint8)


def nearest_component_to_point(bin01: np.ndarray,
                               px: int,
                               py: int,
                               min_area: int,
                               max_area: int,
                               max_dist_px: float) -> Optional[np.ndarray]:
    if bin01.dtype != np.uint8:
        bin01 = bin01.astype(np.uint8)
    H, W = bin01.shape[:2]
    if H <= 0 or W <= 0:
        return None
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if n <= 1:
        return None

    best_lab = 0
    best_d2 = 1e30
    d2_max = float(max_dist_px) * float(max_dist_px)

    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area) or area > int(max_area):
            continue
        cx, cy = centroids[lab]
        d2 = (float(cx) - float(px)) ** 2 + (float(cy) - float(py)) ** 2
        if d2 > d2_max:
            continue
        if d2 < best_d2:
            best_d2 = d2
            best_lab = lab

    if best_lab <= 0:
        return None
    return (labels == best_lab).astype(np.uint8)


def scale_contour(contour: np.ndarray, scale: float) -> np.ndarray:
    if abs(float(scale) - 1.0) < 1e-8:
        return contour.copy()
    pts = contour.astype(np.float32).copy()
    pts[:, 0, 0] *= float(scale)
    pts[:, 0, 1] *= float(scale)
    return np.round(pts).astype(np.int32)


def local_click_outline(arena_bgr: np.ndarray,
                        click_xy: Tuple[int, int],
                        radius: int = 70,
                        min_area: int = 8,
                        max_area_frac: float = 0.35,
                        tiny_mode: bool = False) -> Optional[np.ndarray]:
    H, W = arena_bgr.shape[:2]
    cx, cy = int(click_xy[0]), int(click_xy[1])
    r = int(max(25, radius))

    x0 = int(np.clip(cx - r, 0, W - 1))
    y0 = int(np.clip(cy - r, 0, H - 1))
    x1 = int(np.clip(cx + r, x0 + 1, W))
    y1 = int(np.clip(cy + r, y0 + 1, H))

    patch = arena_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if tiny_mode:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        low = cv2.GaussianBlur(gray, (0, 0), 8.0)
        gray = cv2.addWeighted(gray, 1.0, low, -1.0, 128.0)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.6)
    else:
        gray = cv2.GaussianBlur(gray, (0, 0), 1.2)

    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cand_masks = []
    bin_candidates: List[np.ndarray] = [
        (gray < thr).astype(np.uint8),
        (gray > thr).astype(np.uint8),
    ]
    if tiny_mode:
        ad = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3
        )
        bin_candidates.append((ad > 0).astype(np.uint8))
        bin_candidates.append((ad == 0).astype(np.uint8))

    for bin01 in bin_candidates:
        if tiny_mode:
            bin01 = cv2.morphologyEx(bin01, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        else:
            bin01 = cv2.morphologyEx(bin01, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            bin01 = cv2.morphologyEx(bin01, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        pcx, pcy = cx - x0, cy - y0
        comp = component_containing_point(bin01, pcx, pcy)
        if comp is None and tiny_mode:
            max_area_px = int(max(1, round(max_area_frac * float(bin01.size))))
            comp = nearest_component_to_point(
                bin01,
                pcx,
                pcy,
                min_area=int(min_area),
                max_area=max_area_px,
                max_dist_px=float(max(10, int(round(0.38 * r))))
            )
        if comp is None:
            continue

        area = int(comp.sum())
        if area < int(min_area):
            continue
        if area > int(max_area_frac * comp.size):
            continue

        cand_masks.append(comp)

    if not cand_masks:
        return None

    pcx, pcy = cx - x0, cy - y0

    def comp_score(m: np.ndarray) -> float:
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            return -1e18
        area = float(xs.size)
        d2 = float(np.mean((xs.astype(np.float32) - float(pcx)) ** 2 + (ys.astype(np.float32) - float(pcy)) ** 2))
        if tiny_mode:
           
            return -(0.06 * d2) - (0.0015 * area)
        return area - 0.03 * d2

    comp = max(cand_masks, key=comp_score)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = (comp * 255).astype(np.uint8)

    cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < float(min_area):
        return None
    return cnt


class RunningBGSegmenter:
    def __init__(self,
                 diff_thresh: int = 18,
                 bg_alpha: float = 0.01,
                 blur_sigma: float = 0.0,
                 fg_leak_frac: float = 0.02,
                 tiny_mode: bool = False):
        self.diff_thresh = int(np.clip(diff_thresh, 1, 255))
        self.bg_alpha = float(np.clip(bg_alpha, 0.0001, 0.5))
        self.blur_sigma = float(max(0.0, blur_sigma))
        self.fg_leak_frac = float(np.clip(fg_leak_frac, 0.0, 0.25))
        self.tiny_mode = bool(tiny_mode)
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)) if self.tiny_mode else None
        self._bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) if self.tiny_mode else None
        self.bg_float: Optional[np.ndarray] = None
        self.prev_diff: Optional[np.ndarray] = None

    def preprocess_arena(self, arena_bgr: np.ndarray) -> np.ndarray:
  
        lab = cv2.cvtColor(arena_bgr, cv2.COLOR_BGR2LAB)
        lch = lab[:, :, 0]
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        chroma_u8 = np.clip(chroma * (255.0 / 181.0), 0.0, 255.0).astype(np.uint8)
        gray = cv2.addWeighted(lch, 0.84, chroma_u8, 0.16, 0.0)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
            low = cv2.GaussianBlur(gray, (0, 0), 10.0)
            gray = cv2.addWeighted(gray, 1.0, low, -1.0, 128.0)
        if self._bh_kernel is not None:
   
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._bh_kernel)
            gray = cv2.addWeighted(gray, 0.78, bh, 1.25, 0.0)
        if self.blur_sigma > 0.01:
            gray = cv2.GaussianBlur(gray, (0, 0), self.blur_sigma)
        if self.tiny_mode:
            blur = cv2.GaussianBlur(gray, (0, 0), 0.9)
            gray = cv2.addWeighted(gray, 1.40, blur, -0.40, 0.0)
        return gray

    def initialize_from_gray(self, gray_bg: np.ndarray) -> None:
        self.bg_float = gray_bg.astype(np.float32).copy()
        self.prev_diff = None

    def initialize(self, arena_bgr: np.ndarray) -> None:
        gray = self.preprocess_arena(arena_bgr)
        self.bg_float = gray.astype(np.float32).copy()
        self.prev_diff = None

    def step(self, arena_bgr: np.ndarray) -> np.ndarray:
        if self.bg_float is None:
            self.initialize(arena_bgr)

        gray = self.preprocess_arena(arena_bgr)
        gray_f = gray.astype(np.float32)

        bg = self.bg_float

        drift = float(np.median(gray_f.reshape(-1)) - np.median(bg.reshape(-1)))
        drift = float(np.clip(drift, -24.0, 24.0))
        bg_cmp = bg + drift
        diff = cv2.absdiff(gray_f, bg_cmp)

        motion = diff
        if self.tiny_mode:
            if self.prev_diff is None or self.prev_diff.shape != diff.shape:
                self.prev_diff = diff.copy()
            motion = np.maximum(diff, 0.72 * self.prev_diff + 0.28 * diff)
            self.prev_diff = motion.copy()

        flat = motion.reshape(-1)
        if flat.size > 1_600_000:
            flat_stats = flat[::8]
        elif flat.size > 450_000:
            flat_stats = flat[::4]
        else:
            flat_stats = flat

        med = float(np.median(flat_stats))
        mad = float(np.median(np.abs(flat_stats - med)))
        sigma = float(1.4826 * mad)
        noise_floor = med + max(1.0, 1.00 * sigma)

        if sigma < 1e-6:
            dyn_thr = med + (2.0 if self.tiny_mode else 4.0)
        else:
            dyn_thr = med + ((2.25 if self.tiny_mode else 3.25) * sigma)

        if self.tiny_mode:
            pct_thr = float(np.percentile(flat_stats, 98.5))
            thr = max(noise_floor, min(float(self.diff_thresh), dyn_thr, pct_thr))
        else:
            thr = max(float(self.diff_thresh), dyn_thr)

        fg = (motion >= float(np.clip(thr, 1.0, 255.0))).astype(np.uint8) * 255
        pol_thr = max(2.0, 0.45 * float(self.diff_thresh), 0.80 * noise_floor)
        signed_fg = np.logical_or((bg_cmp - gray_f) >= pol_thr, (gray_f - bg_cmp) >= pol_thr).astype(np.uint8) * 255
        fg = cv2.bitwise_or(fg, signed_fg)

        if self.tiny_mode:
 
            dark_diff = bg_cmp - gray_f
            dark_thr = max(2.0, 0.55 * float(self.diff_thresh), 0.92 * noise_floor)
            dark_fg = (dark_diff >= float(np.clip(dark_thr, 1.0, 255.0))).astype(np.uint8) * 255
            fg = cv2.bitwise_or(fg, dark_fg)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
            fg = cv2.dilate(fg, np.ones((2, 2), np.uint8), iterations=1)
        else:
            fg = cv2.medianBlur(fg, 3)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        bg_mask = (fg == 0).astype(np.float32)
        alpha_bg = float(self.bg_alpha)
        alpha_fg = float(self.bg_alpha) * float(self.fg_leak_frac) * (1.5 if self.tiny_mode else 1.0)
        alpha_map = bg_mask * alpha_bg + (1.0 - bg_mask) * alpha_fg
        bg[:] = (1.0 - alpha_map) * bg + alpha_map * gray_f

        self.bg_float = bg
        return fg


def find_contours_from_fg(fg_mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) >= float(min_area):
            out.append(c)
    return out


def shape_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    try:
        return float(cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0))
    except Exception:
        return 1e9


class ShapeTracker:
    def __init__(self,
                 init_contour: np.ndarray,
                 init_center: Tuple[float, float],
                 search_radius: int = 220,
                 lost_patience_frames: int = 10,
                 tiny_mode: bool = False,
                 box_mode: bool = False):
        self.cx, self.cy = float(init_center[0]), float(init_center[1])
        self.vx, self.vy = 0.0, 0.0

        self.base_search_radius = int(max(20, search_radius))
        self.search_radius = int(max(20, search_radius))
        self.lost_patience_frames = int(max(1, lost_patience_frames))
        self.tiny_mode = bool(tiny_mode)
        self.box_mode = bool(box_mode)

        self.template_contour = init_contour.copy()
        self.template_area = float(max(1.0, cv2.contourArea(init_contour)))
        bx, by, bw, bh = cv2.boundingRect(init_contour)
        self.box_w = int(max(2, bw))
        self.box_h = int(max(2, bh))

        self.last_contour: Optional[np.ndarray] = init_contour
        self.lost_count = 0
        self.last_score = 1.0
        self.last_match_id: Optional[int] = None

  
        self.shape_weight = 0.52 if self.tiny_mode else 1.35
        self.distance_weight = 0.0180 if self.tiny_mode else 0.0060
        self.area_weight = 0.35 if self.tiny_mode else 1.10
        self.area_ratio_weight = 0.55 if self.tiny_mode else 0.34

        self.area_lo = 0.15 * self.template_area
        self.area_hi = 7.00 * self.template_area
        self._update_area_limits(self.template_area)
        self.accept_score = -3.30 if self.tiny_mode else -3.60
        self.max_jump_frac = 0.40 if self.tiny_mode else 0.80
        self.max_speed_px = max(2.0, float(self.base_search_radius) * (0.11 if self.tiny_mode else 0.30))

    def _update_area_limits(self, template_area: float) -> None:
        if self.tiny_mode or template_area <= 30.0:
            self.area_lo = 0.04 * template_area
            self.area_hi = 22.0 * template_area + 10.0
        else:
            self.area_lo = 0.12 * template_area
            self.area_hi = 8.00 * template_area

    def predict(self):
        self.cx += self.vx
        self.cy += self.vy

    def _score(self, c: np.ndarray, use_distance: bool) -> float:
        area = float(cv2.contourArea(c))
        if area < 1:
            return -1e18

    
        if self.lost_count > 0:
            relax = float(min(6, self.lost_count))
            area_lo = self.area_lo * (0.55 if self.tiny_mode else max(0.52, 0.82 - 0.06 * relax))
            area_hi = self.area_hi * (1.0 + (0.38 if self.tiny_mode else 0.30) * relax)
        else:
            area_lo = self.area_lo
            area_hi = self.area_hi
        area_ok = (area_lo <= area <= area_hi)
        area_pen = 0.0 if area_ok else 1.0
        ratio_pen = abs(math.log(max(area, 1e-6) / max(self.template_area, 1e-6)))

        sd = shape_distance(c, self.template_contour)
        if self.box_mode:
            sd = 0.0
        if self.tiny_mode and self.template_area < 25.0:
            sd = 0.45 * sd

        if use_distance:
            ccx, ccy = contour_centroid(c)
            d = math.hypot(ccx - self.cx, ccy - self.cy)
        else:
            d = 0.0

        if self.lost_count > 0:
            if self.tiny_mode:
                relax = max(0.45, 1.0 - 0.10 * float(self.lost_count))
                shape_w = self.shape_weight * relax
                area_w = self.area_weight * (0.65 + 0.35 * relax)
                ratio_w = self.area_ratio_weight * (0.55 + 0.45 * relax)
                dist_w = self.distance_weight * 0.80
            else:
                relax = max(0.40, 1.0 - 0.08 * float(self.lost_count))
                shape_w = self.shape_weight * (0.55 + 0.45 * relax)
                area_w = self.area_weight * (0.55 + 0.45 * relax)
                ratio_w = self.area_ratio_weight * (0.55 + 0.45 * relax)
                dist_w = self.distance_weight * 0.88
        else:
            shape_w = self.shape_weight
            area_w = self.area_weight
            ratio_w = self.area_ratio_weight
            dist_w = self.distance_weight

        score = (
            -(shape_w * sd)
            - (dist_w * d)
            - (area_w * area_pen)
            - (ratio_w * ratio_pen)
        )
        return float(score)

    def _box_contour_at(self, cx: float, cy: float, W: int, H: int) -> np.ndarray:
        half_w = 0.5 * float(self.box_w)
        half_h = 0.5 * float(self.box_h)
        x0 = int(round(cx - half_w))
        y0 = int(round(cy - half_h))
        x1 = int(round(cx + half_w))
        y1 = int(round(cy + half_h))
        x0 = int(np.clip(x0, 0, max(0, W - 2)))
        y0 = int(np.clip(y0, 0, max(0, H - 2)))
        x1 = int(np.clip(max(x0 + 1, x1), x0 + 1, max(1, W - 1)))
        y1 = int(np.clip(max(y0 + 1, y1), y0 + 1, max(1, H - 1)))
        return np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.int32
        )

    def update(self,
               contours: List[np.ndarray],
               W: int,
               H: int,
               contour_ids: Optional[List[Optional[int]]] = None) -> Tuple[bool, Tuple[float, float], Optional[np.ndarray], float]:
        self.predict()
        self.cx = float(np.clip(self.cx, 0, max(0, W - 1)))
        self.cy = float(np.clip(self.cy, 0, max(0, H - 1)))
        pred_x, pred_y = self.cx, self.cy
        if contour_ids is None or len(contour_ids) != len(contours):
            contour_ids = [None] * len(contours)
        self.last_match_id = None

        if not contours:
            self.lost_count += 1
            self.last_score = 0.0
            grow = 0.45 if self.tiny_mode else 0.30
            self.search_radius = min(int(self.base_search_radius * (1.0 + grow * self.lost_count)), 5000)
            return False, (self.cx, self.cy), None, self.last_score

        use_distance = True
        if (not self.tiny_mode) and self.lost_count >= self.lost_patience_frames:
            use_distance = False

        candidates = contours
        candidate_ids = contour_ids
        if use_distance:
            r2 = float(self.search_radius ** 2)
            near = []
            near_ids: List[Optional[int]] = []
            for i, c in enumerate(contours):
                ccx, ccy = contour_centroid(c)
                if (ccx - self.cx) ** 2 + (ccy - self.cy) ** 2 <= r2:
                    near.append(c)
                    near_ids.append(contour_ids[i])
            if not near:
                self.lost_count += 1
                self.last_score = 0.0
                grow = 0.45 if self.tiny_mode else 0.30
                self.search_radius = min(int(self.base_search_radius * (1.0 + grow * self.lost_count)), 5000)
                return False, (self.cx, self.cy), None, self.last_score
            candidates = near
            candidate_ids = near_ids

        best = None
        best_id: Optional[int] = None
        best_s = -1e18
        for i, c in enumerate(candidates):
            s = self._score(c, use_distance=use_distance)
            if s > best_s:
                best_s = s
                best = c
                best_id = candidate_ids[i]

        if best is None:
            self.lost_count += 1
            self.last_score = 0.0
            grow = 0.45 if self.tiny_mode else 0.30
            self.search_radius = min(int(self.base_search_radius * (1.0 + grow * self.lost_count)), 5000)
            return False, (self.cx, self.cy), None, self.last_score

        accept_score = self.accept_score
        if self.lost_count > 0:
            if self.tiny_mode:
                accept_score -= min(1.4, 0.12 * float(self.lost_count))
            else:
                accept_score -= min(1.1, 0.10 * float(self.lost_count))

        if best_s < accept_score:
            self.lost_count += 1
            self.last_score = 0.0
            grow = 0.45 if self.tiny_mode else 0.30
            self.search_radius = min(int(self.base_search_radius * (1.0 + grow * self.lost_count)), 5000)
            return False, (self.cx, self.cy), None, self.last_score

        ccx, ccy = contour_centroid(best)
        area = float(cv2.contourArea(best))
        if self.tiny_mode and area <= 60.0:
            x, y, w, h = cv2.boundingRect(best)
            bcx, bcy = x + 0.5 * w, y + 0.5 * h
            ccx = 0.65 * ccx + 0.35 * bcx
            ccy = 0.65 * ccy + 0.35 * bcy

        jump = math.hypot(ccx - pred_x, ccy - pred_y)
        if self.tiny_mode:
            reacq_boost = 1.0 + 0.20 * float(self.lost_count)
            max_jump = max(
                10.0,
                min(
                    float(self.search_radius) * self.max_jump_frac,
                    (3.5 * float(self.max_speed_px) + 6.0) * reacq_boost,
                ),
            )
        else:
            reacq_boost = 1.0 + 0.15 * float(self.lost_count)
            max_jump = max(
                12.0,
                min(
                    float(self.search_radius) * self.max_jump_frac + 4.0,
                    (2.8 * float(self.max_speed_px) + 10.0) * reacq_boost,
                ),
            )
        if jump > max_jump:
            self.lost_count += 1
            self.last_score = 0.0
            grow = 0.45 if self.tiny_mode else 0.30
            self.search_radius = min(int(self.base_search_radius * (1.0 + grow * self.lost_count)), 5000)
            return False, (self.cx, self.cy), None, self.last_score

        conf = float(np.clip((best_s - accept_score) / 1.6, 0.0, 1.0))
        meas_alpha = (0.20 + 0.26 * conf) if self.tiny_mode else (0.42 + 0.28 * conf)
        nx = (1.0 - meas_alpha) * pred_x + meas_alpha * float(ccx)
        ny = (1.0 - meas_alpha) * pred_y + meas_alpha * float(ccy)

        self.vx = 0.72 * self.vx + 0.28 * (nx - pred_x)
        self.vy = 0.72 * self.vy + 0.28 * (ny - pred_y)
        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed_px and speed > 1e-6:
            scl = self.max_speed_px / speed
            self.vx *= scl
            self.vy *= scl
        self.cx, self.cy = nx, ny

        display_contour = best
        if self.box_mode:
            display_contour = self._box_contour_at(self.cx, self.cy, W, H)

        self.last_contour = display_contour
        self.last_match_id = best_id
        self.lost_count = 0
        self.search_radius = self.base_search_radius

        disp = 1.0 / (1.0 + math.exp(-(best_s + 0.6)))
        self.last_score = float(np.clip(disp, 0.0, 1.0))

        blend = 0.08 if self.tiny_mode else 0.03
        self.template_area = (1.0 - blend) * self.template_area + blend * area
        self._update_area_limits(self.template_area)

        if (not self.box_mode) and self.last_score > (0.45 if self.tiny_mode else 0.55):
            self.template_contour = best.copy()

        return True, (self.cx, self.cy), display_contour, self.last_score


class BoxSelectLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #111; border: 1px solid #333; }")
        self._pixmap: Optional[QPixmap] = None
        self._dragging = False
        self._start = QPoint(0, 0)
        self._end = QPoint(0, 0)
        self._rect = QRect()

    def setPixmap(self, pm: QPixmap) -> None:
        self._pixmap = pm.copy()
        self._update_display_pixmap()
        self._rect = QRect()
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_display_pixmap()

    def _update_display_pixmap(self) -> None:
        if self._pixmap is None:
            super().clear()
            return
        w = max(1, self.width())
        h = max(1, self.height())
        scaled = self._pixmap.scaled(
            w,
            h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def currentRect(self) -> QRect:
        return self._rect

    def clearRect(self) -> None:
        self._rect = QRect()
        self.update()

    def mousePressEvent(self, ev):
        if self._pixmap is None:
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start = ev.position().toPoint()
            self._end = self._start
            self._rect = QRect(self._start, self._end).normalized()
            self.update()

    def mouseMoveEvent(self, ev):
        if self._pixmap is None:
            return
        if self._dragging:
            self._end = ev.position().toPoint()
            self._rect = QRect(self._start, self._end).normalized()
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._pixmap is None:
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._end = ev.position().toPoint()
            self._rect = QRect(self._start, self._end).normalized()
            self.update()
            dx = abs(int(self._end.x()) - int(self._start.x()))
            dy = abs(int(self._end.y()) - int(self._start.y()))
            if dx <= 3 and dy <= 3:
                self.clicked.emit(self._end)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self._rect.isNull() and self._rect.width() > 5 and self._rect.height() > 5:
            from PyQt6.QtGui import QPainter, QPen
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.cyan)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self._rect)

    def rect_to_image_coords(self, rect: QRect, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        if self._pixmap is None or rect.isNull():
            return (0, 0, img_w, img_h)

        label_w, label_h = self.width(), self.height()
        label_w = max(1, int(label_w))
        label_h = max(1, int(label_h))

        rx1 = int(np.clip(rect.left(), 0, label_w - 1))
        ry1 = int(np.clip(rect.top(), 0, label_h - 1))
        rx2 = int(np.clip(rect.right(), 0, label_w - 1))
        ry2 = int(np.clip(rect.bottom(), 0, label_h - 1))

        px1 = (float(rx1) / float(label_w)) * float(img_w)
        py1 = (float(ry1) / float(label_h)) * float(img_h)
        px2 = (float(rx2) / float(label_w)) * float(img_w)
        py2 = (float(ry2) / float(label_h)) * float(img_h)

        ix1 = int(np.clip(px1, 0, img_w - 1))
        iy1 = int(np.clip(py1, 0, img_h - 1))
        ix2 = int(np.clip(px2, 0, img_w - 1))
        iy2 = int(np.clip(py2, 0, img_h - 1))

        x = min(ix1, ix2)
        y = min(iy1, iy2)
        w = max(2, abs(ix2 - ix1))
        h = max(2, abs(iy2 - iy1))
        return (x, y, w, h)

    def point_to_image_coords(self, pt: QPoint, img_w: int, img_h: int) -> Tuple[int, int]:
        if self._pixmap is None:
            return (0, 0)

        label_w, label_h = self.width(), self.height()
        label_w = max(1, int(label_w))
        label_h = max(1, int(label_h))

        x = (float(np.clip(pt.x(), 0, label_w - 1)) / float(label_w)) * float(img_w)
        y = (float(np.clip(pt.y(), 0, label_h - 1)) / float(label_h)) * float(img_h)
        ix = int(np.clip(x, 0, img_w - 1))
        iy = int(np.clip(y, 0, img_h - 1))
        return ix, iy


@dataclass
class TargetSpec:
    tid: int
    click_x: int
    click_y: int
    init_contour_xy: List[List[int]]
    box_mode: bool = False


@dataclass
class SessionConfig:
    created_at: str
    session_name: str
    video_path: str
    output_dir: str

    arena_x: int = 0
    arena_y: int = 0
    arena_w: int = 0
    arena_h: int = 0

    targets: List[TargetSpec] = None

    tiny_mode: bool = True
    processing_scale: float = 2.0

    search_radius: int = 220
    lost_patience_frames: int = 10

    diff_thresh: int = 18
    bg_alpha: float = 0.01
    min_blob_area: int = 8
    seg_blur_sigma: float = 0.0

    heat_sigma: float = 10.0
    heat_post_blur: float = 0.0
    clip_percentile: float = 99.0
    gamma: float = 0.8
    overlay_alpha: float = 0.45

    export_overlay_video: bool = True
    overlay_video_fps: float = 30.0
    show_live_tracking: bool = False

    warmup_frames: int = 25
    stats_grid_cols: int = 10
    stats_grid_rows: int = 10
    stats_alpha: float = 0.05
    stats_mc_iterations: int = 4000
    stats_random_seed: int = 2026

    def __post_init__(self):
        if self.targets is None:
            self.targets = []


class ProcessWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    live_frame = pyqtSignal(QImage)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SessionConfig, quick_test_seconds: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.quick_test_seconds = float(quick_test_seconds)

    def run(self):
        try:
            self._process()
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n\n{tb}")

    def _process(self):
        cfg = self.cfg
        video_path = Path(cfg.video_path)
        out_dir = Path(cfg.output_dir)

        in_dir = out_dir / "input"
        cfg_dir = out_dir / "config"
        out_outputs = out_dir / "outputs"
        log_dir = out_dir / "logs"
        safe_mkdir(in_dir)
        safe_mkdir(cfg_dir)
        safe_mkdir(out_outputs)
        safe_mkdir(log_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if self.quick_test_seconds > 0:
            max_frames = int(self.quick_test_seconds * fps)
        else:
            max_frames = total_frames if total_frames > 0 else None

        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            raise RuntimeError("Could not read first frame.")

        ax, ay, aw, ah = cfg.arena_x, cfg.arena_y, cfg.arena_w, cfg.arena_h
        if aw <= 0 or ah <= 0:
            ax, ay, aw, ah = 0, 0, VW, VH
        ax, ay, aw, ah = clamp_xywh(ax, ay, aw, ah, VW, VH)
        proc_scale = float(max(1.0, cfg.processing_scale))
        if not cfg.tiny_mode:
            proc_scale = 1.0
        proc_w = int(max(2, round(aw * proc_scale)))
        proc_h = int(max(2, round(ah * proc_scale)))

        arena0 = frame0[ay:ay + ah, ax:ax + aw].copy()

        def arena_for_processing(fr: np.ndarray) -> np.ndarray:
            ar = fr[ay:ay + ah, ax:ax + aw]
            if abs(proc_scale - 1.0) < 1e-8:
                return ar
            return cv2.resize(ar, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)

        if not cfg.targets:
            cap.release()
            raise RuntimeError("No targets added. Define Target then Add Target at least once.")

        warm_frames: List[np.ndarray] = [frame0]
        warm_n = int(max(1, cfg.warmup_frames))
        for _ in range(max(0, warm_n - 1)):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            warm_frames.append(fr)

        seg_tmp = RunningBGSegmenter(
            diff_thresh=int(cfg.diff_thresh),
            bg_alpha=float(cfg.bg_alpha),
            blur_sigma=float(cfg.seg_blur_sigma),
            fg_leak_frac=0.02,
            tiny_mode=bool(cfg.tiny_mode),
        )

        gray_stack = []
        for fr in warm_frames:
            ar = arena_for_processing(fr)
            g = seg_tmp.preprocess_arena(ar)
            gray_stack.append(g)

        gray_bg = np.median(np.stack(gray_stack, axis=0), axis=0).astype(np.uint8)
        seg_tmp.initialize_from_gray(gray_bg)

        segmenter = seg_tmp

        trackers: Dict[int, ShapeTracker] = {}
        centers: Dict[int, List[Tuple[float, float]]] = {t.tid: [] for t in cfg.targets}
        heats: Dict[int, np.ndarray] = {t.tid: np.zeros((ah, aw), dtype=np.float32) for t in cfg.targets}

        for t in cfg.targets:
            pts = np.array(t.init_contour_xy, dtype=np.int32).reshape((-1, 1, 2))
            pts_proc = scale_contour(pts, proc_scale)
            cx, cy = contour_centroid(pts_proc)
            trackers[t.tid] = ShapeTracker(
                init_contour=pts_proc,
                init_center=(cx, cy),
                search_radius=int(round(cfg.search_radius * proc_scale)),
                lost_patience_frames=int(cfg.lost_patience_frames),
                tiny_mode=bool(cfg.tiny_mode),
                box_mode=bool(getattr(t, "box_mode", False)),
            )

        overlay_writer = None
        overlay_path = out_outputs / "overlay_multi.mp4"
        if cfg.export_overlay_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = float(cfg.overlay_video_fps) if cfg.overlay_video_fps > 1 else float(fps)
            overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, out_fps, (aw, ah))

        tracks_path = out_outputs / "tracks_multi.csv"
        fcsv = open(tracks_path, "w", encoding="utf-8")
        fcsv.write("frame,time_s,target_id,x,y,ok,score\n")
        render_live = bool(cfg.show_live_tracking)
        render_overlay = overlay_writer is not None
        render_vis = bool(render_live or render_overlay)
        live_stride = 3

        frame_idx = 0

        def process_frame(fr: np.ndarray, idx: int):
            arena = fr[ay:ay + ah, ax:ax + aw]
            proc_arena = arena_for_processing(fr)
            vis = arena.copy() if render_vis else None
            t_s = idx / float(fps)

            fg = segmenter.step(proc_arena)
            contours = find_contours_from_fg(fg, min_area=int(cfg.min_blob_area))

            used_contour_ids: set[int] = set()
            matched_centers_proc: List[Tuple[float, float]] = []
            tracker_items = sorted(trackers.items(), key=lambda kv: (kv[1].lost_count, kv[0]))

            for tid, trk in tracker_items:
                sep_px = float(max(10.0, 0.60 * math.sqrt(max(1.0, float(trk.template_area)))))

                def is_far_from_matched(cx: float, cy: float) -> bool:
                    for mx, my in matched_centers_proc:
                        if (float(cx) - float(mx)) ** 2 + (float(cy) - float(my)) ** 2 < (sep_px * sep_px):
                            return False
                    return True

                base_pool: List[Tuple[int, np.ndarray]] = [
                    (cid, c)
                    for cid, c in enumerate(contours)
                    if (cid not in used_contour_ids) and is_far_from_matched(*contour_centroid(c))
                ]
                cand_contours = [c for _, c in base_pool]
                cand_ids: List[Optional[int]] = [cid for cid, _ in base_pool]
                pred_x = int(np.clip(round(trk.cx + trk.vx), 0, max(0, proc_w - 1)))
                pred_y = int(np.clip(round(trk.cy + trk.vy), 0, max(0, proc_h - 1)))
                if cfg.tiny_mode:
                    local_radius = int(max(26, min(140, round(0.35 * trk.search_radius))))
                    local_min_area = max(1, int(round(cfg.min_blob_area * 0.5)))
                    local_max_area_frac = 0.18
                    tight_r = float(max(22.0, 0.42 * float(trk.search_radius)))
                    area_lo = 0.25 * float(trk.area_lo)
                    area_hi = 1.80 * float(trk.area_hi)
                else:
                    local_radius = int(max(34, min(220, round(0.42 * trk.search_radius))))
                    local_min_area = max(1, int(round(cfg.min_blob_area * 0.7)))
                    local_max_area_frac = 0.24
                    tight_r = float(max(28.0, 0.55 * float(trk.search_radius)))
                    area_lo = 0.40 * float(trk.area_lo)
                    area_hi = 1.90 * float(trk.area_hi)
                tight_r2 = tight_r * tight_r

                near_contours: List[np.ndarray] = []
                near_ids: List[Optional[int]] = []
                for cid, c in base_pool:
                    a = float(cv2.contourArea(c))
                    if not (area_lo <= a <= area_hi):
                        continue
                    ccx, ccy = contour_centroid(c)
                    d2 = (float(ccx) - float(pred_x)) ** 2 + (float(ccy) - float(pred_y)) ** 2
                    if d2 <= tight_r2:
                        near_contours.append(c)
                        near_ids.append(cid)

                if near_contours:
                    cand_contours = near_contours
                    cand_ids = near_ids

                run_local_reacq = bool(cfg.tiny_mode or trk.lost_count > 0 or len(cand_contours) < 2)
                if run_local_reacq:
                    local_c = local_click_outline(
                        proc_arena,
                        (pred_x, pred_y),
                        radius=local_radius,
                        min_area=local_min_area,
                        max_area_frac=local_max_area_frac,
                        tiny_mode=bool(cfg.tiny_mode),
                    )
                    if local_c is not None:
                        lcx, lcy = contour_centroid(local_c)
                        if not is_far_from_matched(lcx, lcy):
                            local_c = None
                    if local_c is not None:
                        local_area = float(cv2.contourArea(local_c))
                        local_ok = (0.35 * float(trk.area_lo) <= local_area <= 1.80 * float(trk.area_hi))
                        if local_ok:
      
                            cand_contours = [local_c] + cand_contours
                            cand_ids = [None] + cand_ids

                ok_tr, (pcx, pcy), contour_proc, score = trk.update(
                    cand_contours,
                    proc_w,
                    proc_h,
                    contour_ids=cand_ids,
                )
                cx = float(np.clip(pcx / proc_scale, 0, max(0, aw - 1)))
                cy = float(np.clip(pcy / proc_scale, 0, max(0, ah - 1)))
                if ok_tr and trk.last_match_id is not None:
                    used_contour_ids.add(int(trk.last_match_id))
                if ok_tr:
                    matched_centers_proc.append((float(pcx), float(pcy)))

                if ok_tr:
                    centers[tid].append((cx, cy))
                    gaussian_splat(heats[tid], cx, cy, float(cfg.heat_sigma), weight=1.0)

                if render_vis and vis is not None:
                    if contour_proc is not None:
                        contour = scale_contour(contour_proc, 1.0 / proc_scale)
                        cv2.drawContours(vis, [contour], -1, (0, 255, 255) if ok_tr else (0, 0, 255), 2)
                    else:
                        if trk.last_contour is not None:
                            last_contour = scale_contour(trk.last_contour, 1.0 / proc_scale)
                            cv2.drawContours(vis, [last_contour], -1, (70, 70, 255), 1)

                    cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0) if ok_tr else (0, 0, 255), -1)
                    cv2.putText(
                        vis, f"{tid} {'OK' if ok_tr else 'LOST'} {score:.3f}",
                        (int(cx) + 6, int(max(0, cy - 6))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if ok_tr else (0, 0, 255), 2
                    )

                    tail = centers[tid]
                    tail_n = min(120, len(tail))
                    for i in range(len(tail) - tail_n, len(tail) - 1):
                        x1, y1 = tail[i]
                        x2, y2 = tail[i + 1]
                        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

                fcsv.write(f"{idx},{t_s:.6f},{tid},{cx:.3f},{cy:.3f},{int(ok_tr)},{score:.6f}\n")

            if idx % 120 == 0:
                fcsv.flush()

            if render_overlay and vis is not None:
                overlay_writer.write(vis)

            if render_live and vis is not None and idx % live_stride == 0:
                self.live_frame.emit(cv_to_qimage(vis))
        try:
            for fr in warm_frames:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                process_frame(fr, frame_idx)
                frame_idx += 1
                denom = max_frames if (max_frames is not None and max_frames > 0) else max(total_frames, frame_idx, 1)
                pct = int(np.clip(100 * frame_idx / max(denom, 1), 0, 100))
                if frame_idx % 10 == 0:
                    self.progress.emit(pct)
                    self.status.emit(f"Tracking... frame {frame_idx} ({pct}%) | scale={proc_scale:.1f}x")

            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                process_frame(frame, frame_idx)
                frame_idx += 1
                denom = max_frames if (max_frames is not None and max_frames > 0) else max(total_frames, frame_idx, 1)
                pct = int(np.clip(100 * frame_idx / max(denom, 1), 0, 100))
                if frame_idx % 10 == 0:
                    self.progress.emit(pct)
                    self.status.emit(f"Tracking... frame {frame_idx} ({pct}%) | scale={proc_scale:.1f}x")
        finally:
            fcsv.close()

        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()

        combined = np.zeros((ah, aw), dtype=np.float32)
        for tid, heat in heats.items():
            hm = heat
            if cfg.heat_post_blur and cfg.heat_post_blur > 0.01:
                hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=float(cfg.heat_post_blur), sigmaY=float(cfg.heat_post_blur))
            combined += hm
            heat_clip = percentile_clip(hm, cfg.clip_percentile)
            maxv = float(np.max(heat_clip)) if heat_clip.size else 0.0
            hm01 = heat_clip / maxv if maxv > 0 else heat_clip
            hm01 = apply_gamma(hm01, cfg.gamma)
            colored = colormap_heatmap(hm01)
            cv2.imwrite(str(out_outputs / f"heatmap_target_{tid:03d}.png"), colored)
            alpha = float(np.clip(cfg.overlay_alpha, 0.0, 1.0))
            overlay = cv2.addWeighted(arena0, 1.0 - alpha, colored, alpha, 0.0)
            cv2.imwrite(str(out_outputs / f"heatmap_target_{tid:03d}_overlay.png"), overlay)

        if cfg.heat_post_blur and cfg.heat_post_blur > 0.01:
            combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=float(cfg.heat_post_blur), sigmaY=float(cfg.heat_post_blur))

        comb_clip = percentile_clip(combined, cfg.clip_percentile)
        maxv = float(np.max(comb_clip)) if comb_clip.size else 0.0
        comb01 = comb_clip / maxv if maxv > 0 else comb_clip
        comb01 = apply_gamma(comb01, cfg.gamma)
        comb_col = colormap_heatmap(comb01)
        cv2.imwrite(str(out_outputs / "heatmap_combined.png"), comb_col)
        alpha = float(np.clip(cfg.overlay_alpha, 0.0, 1.0))
        comb_overlay = cv2.addWeighted(arena0, 1.0 - alpha, comb_col, alpha, 0.0)
        cv2.imwrite(str(out_outputs / "heatmap_combined_overlay.png"), comb_overlay)

        self.status.emit("Computing under-exploration statistics...")
        stats_rows, z_grid = underexplored_analysis_grid(
            centers=centers,
            arena_w=aw,
            arena_h=ah,
            grid_cols=int(cfg.stats_grid_cols),
            grid_rows=int(cfg.stats_grid_rows),
            alpha=float(cfg.stats_alpha),
            mc_iterations=int(cfg.stats_mc_iterations),
            random_seed=int(cfg.stats_random_seed),
        )

        under_map = np.maximum(0.0, -z_grid).astype(np.float32)
        if under_map.size > 0 and float(np.max(under_map)) > 0:
            under01 = under_map / float(np.max(under_map))
        else:
            under01 = under_map
     
        under01_up = cv2.resize(under01, (aw, ah), interpolation=cv2.INTER_NEAREST)
        under_vis = colormap_heatmap(under01_up)
    
        under_vis = cv2.addWeighted(
            under_vis, 0.5, np.full_like(under_vis, 255, dtype=np.uint8), 0.5, 0.0
        )
        under_map_vis = under_vis.copy()

        for gy in range(int(cfg.stats_grid_rows) + 1):
            y = int(round(gy * ah / max(int(cfg.stats_grid_rows), 1)))
            cv2.line(under_map_vis, (0, y), (aw - 1, y), (220, 220, 220), 1)
        for gx in range(int(cfg.stats_grid_cols) + 1):
            x = int(round(gx * aw / max(int(cfg.stats_grid_cols), 1)))
            cv2.line(under_map_vis, (x, 0), (x, ah - 1), (220, 220, 220), 1)

        under_cells = [r for r in stats_rows if int(r["is_underexplored"]) == 1]
        for r in under_cells:
            x0, y0 = int(r["x0"]), int(r["y0"])
            x1, y1 = int(r["x1"]) - 1, int(r["y1"]) - 1
            cv2.rectangle(under_map_vis, (x0, y0), (x1, y1), (0, 0, 255), 2)

        cv2.imwrite(str(out_outputs / "underexplored_map.png"), under_map_vis)

        stats_csv = out_outputs / "underexplored_stats.csv"
        with open(stats_csv, "w", encoding="utf-8") as fstats:
            fstats.write(
                "cell_row,cell_col,x0,y0,x1,y1,observed,expected,null_prob,z_score,p_value_low_tail,p_value_fdr,is_underexplored\n"
            )
            for row in stats_rows:
                fstats.write(
                    f"{int(row['cell_row'])},{int(row['cell_col'])},{int(row['x0'])},{int(row['y0'])},"
                    f"{int(row['x1'])},{int(row['y1'])},{int(row['observed'])},{float(row['expected']):.6f},"
                    f"{float(row['null_prob']):.8f},{float(row['z_score']):.6f},{float(row['p_value_low_tail']):.6g},"
                    f"{float(row['p_value_fdr']):.6g},{int(row['is_underexplored'])}\n"
                )

        cfg_path = cfg_dir / "session.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        self.progress.emit(100)
        self.status.emit(f"Done! Saved to: {out_dir}")
        self.finished_ok.emit(str(out_dir))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINtrk: Multi-Target Contour-Based Tracking with Gaussian-Kernel Occupancy Heatmap Generation")
        self.setAcceptDrops(True)

        self.video_path: Optional[Path] = None
        self.first_frame_bgr: Optional[np.ndarray] = None
        self.first_frame_pixmap: Optional[QPixmap] = None

        self.mode: str = "idle"
        self.arena_full: Optional[Tuple[int, int, int, int]] = None

        self._pending_click_full: Optional[Tuple[int, int]] = None
        self._pending_contour_rel: Optional[np.ndarray] = None
        self._pending_box_mode: bool = False

        self.targets_full: List[TargetSpec] = []
        self.next_tid: int = 1

        self.worker: Optional[ProcessWorker] = None

        open_action = QAction("Import Video...", self)
        open_action.triggered.connect(self.import_video)
        self.menuBar().addAction(open_action)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self.preview = BoxSelectLabel()
        self.preview.clicked.connect(self._on_preview_clicked)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Import Video…")
        self.btn_import.clicked.connect(self.import_video)

        self.btn_define_arena = QPushButton("Define Arena")
        self.btn_define_arena.clicked.connect(self.on_define_arena)

        self.btn_full_arena = QPushButton("Use Full Frame Arena")
        self.btn_full_arena.clicked.connect(self.on_full_frame_arena)

        self.btn_save_arena = QPushButton("Save Arena")
        self.btn_save_arena.clicked.connect(self.on_save_arena)

        self.btn_define_target = QPushButton("Define Target")
        self.btn_define_target.clicked.connect(self.on_define_target_mode)

        self.btn_add_target = QPushButton("Add Target")
        self.btn_add_target.clicked.connect(self.on_add_target_commit)

        self.btn_remove_last = QPushButton("Remove Last Target")
        self.btn_remove_last.clicked.connect(self.on_remove_last)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.on_clear)

        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_define_arena)
        btn_row.addWidget(self.btn_full_arena)
        btn_row.addWidget(self.btn_save_arena)
        btn_row.addWidget(self.btn_define_target)
        btn_row.addWidget(self.btn_add_target)
        btn_row.addWidget(self.btn_remove_last)
        btn_row.addWidget(self.btn_clear)

        self.info_label = QLabel(
            "Flow:\n"
            "1) Define Arena → draw box → Save Arena  (or Use Full Frame Arena)\n"
            "2) Define Target → click target OR draw box → Add Target (repeat)\n"
            "3) Quick Test / RUN\n"
        )
        self.info_label.setWordWrap(True)

        self.targets_list = QListWidget()
        self.targets_list.setStyleSheet("QListWidget { background: #111; color: #eee; border: 1px solid #333; }")


        left.addWidget(self.preview, 5)
        left.addLayout(btn_row)
        left.addWidget(QLabel("Targets:"))
        left.addWidget(self.targets_list, 1)
        left.addWidget(self.info_label)

        right = QVBoxLayout()

        self.session_box = QGroupBox("Session / Run")
        session_form = QFormLayout(self.session_box)

        self.session_name = QLineEdit("")
        self.output_root = QLineEdit(str(Path.home() / "AntHeatmaps"))

        self.quick_seconds = QDoubleSpinBox()
        self.quick_seconds.setRange(1.0, 600.0)
        self.quick_seconds.setValue(5.0)

        self.btn_quick = QPushButton("Quick Test")
        self.btn_quick.clicked.connect(self.on_quick_run)

        self.btn_run = QPushButton("RUN Full Video")
        self.btn_run.clicked.connect(self.on_full_run)
        self.btn_run.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; }")

        self.export_overlay = QCheckBox("Export overlay_multi.mp4")
        self.export_overlay.setChecked(True)

        self.show_live_tracking = QCheckBox("Show live tracking during processing (slower)")
        self.show_live_tracking.setChecked(False)

        self.overlay_fps = QDoubleSpinBox()
        self.overlay_fps.setRange(1.0, 240.0)
        self.overlay_fps.setValue(30.0)

        self.search_radius = QSpinBox()
        self.search_radius.setRange(20, 4000)
        self.search_radius.setValue(180)

        self.lost_patience = QSpinBox()
        self.lost_patience.setRange(1, 240)
        self.lost_patience.setValue(20)

        self.diff_thresh = QSpinBox()
        self.diff_thresh.setRange(1, 255)
        self.diff_thresh.setValue(12)

        self.bg_alpha = QDoubleSpinBox()
        self.bg_alpha.setRange(0.0001, 0.2)
        self.bg_alpha.setSingleStep(0.001)
        self.bg_alpha.setValue(0.01)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 20000)
        self.min_area.setValue(2)

        self.seg_blur = QDoubleSpinBox()
        self.seg_blur.setRange(0.0, 10.0)
        self.seg_blur.setValue(0.0)

        self.tiny_mode = QCheckBox("Tiny / low-quality mode")
        self.tiny_mode.setChecked(True)

        self.processing_scale = QDoubleSpinBox()
        self.processing_scale.setRange(1.0, 4.0)
        self.processing_scale.setSingleStep(0.1)
        self.processing_scale.setValue(2.0)

        self.heat_sigma = QDoubleSpinBox()
        self.heat_sigma.setRange(0.1, 200.0)
        self.heat_sigma.setValue(10.0)

        self.heat_post_blur = QDoubleSpinBox()
        self.heat_post_blur.setRange(0.0, 200.0)
        self.heat_post_blur.setValue(0.0)

        self.clip_pct = QDoubleSpinBox()
        self.clip_pct.setRange(50.0, 100.0)
        self.clip_pct.setValue(99.0)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 3.0)
        self.gamma.setValue(0.8)

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setValue(0.45)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)

        session_form.addRow("Session name", self.session_name)
        session_form.addRow("Output root", self.output_root)
        session_form.addRow("Overlay FPS", self.overlay_fps)
        session_form.addRow("Quick seconds", self.quick_seconds)
        session_form.addRow(self.export_overlay)
        session_form.addRow(self.show_live_tracking)

        session_form.addRow("Search radius (px)", self.search_radius)
        session_form.addRow("Lost patience (frames)", self.lost_patience)

        session_form.addRow(self.tiny_mode)
        session_form.addRow("Processing scale", self.processing_scale)
        session_form.addRow("Diff threshold", self.diff_thresh)
        session_form.addRow("BG update alpha", self.bg_alpha)
        session_form.addRow("Min blob area", self.min_area)
        session_form.addRow("Seg blur sigma", self.seg_blur)

        session_form.addRow("Heat sigma", self.heat_sigma)
        session_form.addRow("Post blur", self.heat_post_blur)
        session_form.addRow("Clip percentile", self.clip_pct)
        session_form.addRow("Gamma", self.gamma)
        session_form.addRow("Overlay alpha", self.alpha)

        session_form.addRow(self.btn_quick)
        session_form.addRow(self.btn_run)
        session_form.addRow(self.progress)
        session_form.addRow(self.status_label)

        right.addWidget(self.session_box)
        right.addStretch(1)

        main_layout.addLayout(left, 3)
        main_layout.addLayout(right, 2)

        self._apply_dark_theme()
        self._fit_to_screen()
        self._sync_buttons()

    def _fit_to_screen(self):
        app = QApplication.instance()
        if app is None:
            return
        screen = app.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        w = int(min(g.width() * 0.92, 1180))
        h = int(min(g.height() * 0.92, 860))
        w = max(900, w)
        h = max(650, h)
        self.resize(w, h)
        self.move(g.x() + (g.width() - w) // 2, g.y() + (g.height() - h) // 2)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0b0b0b; }
            QGroupBox { color: #ddd; border: 1px solid #333; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QLabel { color: #ddd; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #111; color: #eee; border: 1px solid #333; padding: 4px;
            }
            QPushButton {
                background: #1b1b1b; color: #eee; border: 1px solid #333; padding: 8px;
            }
            QPushButton:hover { background: #242424; }
            QPushButton:disabled { color: #666; background: #111; }
            QProgressBar { background: #111; border: 1px solid #333; color: #eee; }
            QProgressBar::chunk { background: #2a7; }
            QCheckBox { color: #ddd; }
        """)

    def _sync_buttons(self):
        has_video = self.first_frame_bgr is not None
        has_arena = self.arena_full is not None
        has_pending = (self._pending_contour_rel is not None and self._pending_click_full is not None)

        self.btn_define_arena.setEnabled(has_video)
        self.btn_full_arena.setEnabled(has_video)
        self.btn_save_arena.setEnabled(has_video and (self.mode == "arena"))

        self.btn_define_target.setEnabled(has_video and has_arena)
        self.btn_add_target.setEnabled(has_video and has_arena and (has_pending or self.mode == "target"))

        self.btn_remove_last.setEnabled(len(self.targets_full) > 0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                p = Path(urls[0].toLocalFile())
                if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".m4v"]:
                    event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        self.load_video(Path(urls[0].toLocalFile()))

    def import_video(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select video",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)"
        )
        if not fn:
            return
        self.load_video(Path(fn))

    def load_video(self, path: Path):
        if not path.exists():
            QMessageBox.warning(self, "Not found", "Video file does not exist.")
            return
        self.video_path = path
        self.session_name.setText(path.stem)

        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "Error", "Could not read first frame.")
            return

        self.first_frame_bgr = frame
        pm = QPixmap.fromImage(cv_to_qimage(frame))
        self.first_frame_pixmap = pm
        self.preview.setPixmap(pm)
        self.preview.clearRect()

        self.mode = "idle"
        self.arena_full = None
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self.targets_full = []
        self.next_tid = 1
        self.targets_list.clear()

        self.status_label.setText("Loaded. Click Define Arena → draw box → Save Arena (or Use Full Frame Arena).")
        self._sync_buttons()

    def _get_box_from_preview(self) -> Tuple[int, int, int, int]:
        if self.first_frame_bgr is None:
            raise RuntimeError("No video loaded.")
        img_h, img_w = self.first_frame_bgr.shape[:2]
        rect = self.preview.currentRect()
        if rect.isNull() or rect.width() < 10 or rect.height() < 10:
            raise RuntimeError("No box drawn. Drag a box on the preview.")
        x, y, w, h = self.preview.rect_to_image_coords(rect, img_w, img_h)
        x, y, w, h = clamp_xywh(x, y, w, h, img_w, img_h)
        return x, y, w, h

    def _has_valid_target_box(self) -> bool:
        if self.first_frame_bgr is None:
            return False
        rect = self.preview.currentRect()
        return (not rect.isNull()) and rect.width() >= 4 and rect.height() >= 4

    def _set_pending_from_drawn_box(self) -> bool:
        if self.first_frame_bgr is None or self.arena_full is None:
            return False
        if not self._has_valid_target_box():
            return False

        img_h, img_w = self.first_frame_bgr.shape[:2]
        rect = self.preview.currentRect()
        x, y, w, h = self.preview.rect_to_image_coords(rect, img_w, img_h)
        x, y, w, h = clamp_xywh(x, y, w, h, img_w, img_h)

        ax, ay, aw, ah = self.arena_full
        x0 = max(x, ax)
        y0 = max(y, ay)
        x1 = min(x + w, ax + aw)
        y1 = min(y + h, ay + ah)
        bw = int(x1 - x0)
        bh = int(y1 - y0)
        if bw < 2 or bh < 2:
            return False

        rx0 = int(x0 - ax)
        ry0 = int(y0 - ay)
        rx1 = int(rx0 + bw - 1)
        ry1 = int(ry0 + bh - 1)
        cx_full = int(x0 + bw // 2)
        cy_full = int(y0 + bh // 2)
        contour = np.array(
            [[[rx0, ry0]], [[rx1, ry0]], [[rx1, ry1]], [[rx0, ry1]]],
            dtype=np.int32
        )

        self._pending_click_full = (cx_full, cy_full)
        self._pending_contour_rel = contour
        self._pending_box_mode = True

        contour_full = contour.copy()
        contour_full[:, 0, 0] += ax
        contour_full[:, 0, 1] += ay
        self._reset_preview_pixmap()
        self._draw_contour_on_preview(contour_full)
        return True

    def on_define_arena(self):
        if self.first_frame_bgr is None:
            return
        self.mode = "arena"
        self.preview.clearRect()
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self._reset_preview_pixmap()
        self.status_label.setText("Arena: draw a box, then click Save Arena.")
        self._sync_buttons()

    def on_full_frame_arena(self):
        if self.first_frame_bgr is None:
            return
        h, w = self.first_frame_bgr.shape[:2]
        self.arena_full = (0, 0, w, h)
        self.mode = "idle"
        self.preview.clearRect()
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self._reset_preview_pixmap()
        self.status_label.setText("Arena set to full frame. Now Define Target → click target or draw box → Add Target.")
        self._sync_buttons()

    def on_save_arena(self):
        try:
            x, y, w, h = self._get_box_from_preview()
        except Exception as e:
            QMessageBox.warning(self, "Arena error", str(e))
            return
        self.arena_full = (x, y, w, h)
        self.mode = "idle"
        self.preview.clearRect()
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self._reset_preview_pixmap()
        self.status_label.setText("Arena saved. Now Define Target → click target or draw box → Add Target.")
        self._sync_buttons()

    def on_define_target_mode(self):
        if self.first_frame_bgr is None:
            return
        if self.arena_full is None:
            QMessageBox.information(self, "Arena not set", "Define and Save Arena first.")
            return
        self.mode = "target"
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self.preview.clearRect()
        self._reset_preview_pixmap()
        self.status_label.setText(
            "Target mode: click target for auto-outline, or drag a box around target, then click Add Target."
        )
        self._sync_buttons()

    def _reset_preview_pixmap(self):
        if self.first_frame_pixmap is not None:
            self.preview.setPixmap(self.first_frame_pixmap)

    def _draw_contour_on_preview(self, contour_full: np.ndarray):
        if self.first_frame_bgr is None or self.first_frame_pixmap is None:
            return
        pm = QPixmap(self.first_frame_pixmap)
        from PyQt6.QtGui import QPainter, QPen

        painter = QPainter(pm)
        pen = QPen(Qt.GlobalColor.cyan)
        pen.setWidth(3)
        painter.setPen(pen)

        pts = contour_full.reshape(-1, 2)
        if pts.shape[0] >= 2:
            qpts = []
            for (x, y) in pts:
                qpts.append(QPoint(int(x), int(y)))
            for i in range(len(qpts)):
                painter.drawLine(qpts[i], qpts[(i + 1) % len(qpts)])

        painter.end()
        self.preview.setPixmap(pm)

    def _on_preview_clicked(self, pt: QPoint):
        if self.mode != "target":
            return
        if self.first_frame_bgr is None or self.arena_full is None:
            return

        img_h, img_w = self.first_frame_bgr.shape[:2]
        ix, iy = self.preview.point_to_image_coords(pt, img_w, img_h)

        ax, ay, aw, ah = self.arena_full
        if not (ax <= ix < ax + aw and ay <= iy < ay + ah):
            self.status_label.setText("Click inside the arena to define a target.")
            return

        local = (ix - ax, iy - ay)
        arena0 = self.first_frame_bgr[ay:ay + ah, ax:ax + aw].copy()

        tiny_mode = bool(self.tiny_mode.isChecked())
        click_radius = 95 if tiny_mode else 70
        click_min_area = max(1, int(round(self.min_area.value() * (0.5 if tiny_mode else 1.0))))
        contour = local_click_outline(
            arena0,
            local,
            radius=click_radius,
            min_area=click_min_area,
            max_area_frac=0.20 if tiny_mode else 0.35,
            tiny_mode=tiny_mode,
        )
        if contour is None:
            QMessageBox.warning(
                self,
                "Target error",
                "Could not capture an outline from that click.\n\n"
                "Try clicking closer to the center of the ant.\n"
                "For tiny/noisy footage, keep Tiny / low-quality mode enabled,\n"
                "lower Min blob area, and lower Diff threshold."
            )
            return

        self._pending_click_full = (ix, iy)
        self._pending_contour_rel = contour.copy()
        self._pending_box_mode = False

        contour_full = contour.copy()
        contour_full[:, 0, 0] += ax
        contour_full[:, 0, 1] += ay

        self._reset_preview_pixmap()
        self._draw_contour_on_preview(contour_full)

        area = cv2.contourArea(contour)
        self.status_label.setText(f"Captured outline (area={area:.1f}). Click Add Target to commit.")
        self._sync_buttons()

    def on_add_target_commit(self):
        if self._pending_click_full is None or self._pending_contour_rel is None:
            used_box = self._set_pending_from_drawn_box()
            if not used_box:
                QMessageBox.information(
                    self,
                    "Target not set",
                    "Click a target or draw a box around it, then click Add Target."
                )
                return

        tid = self.next_tid
        self.next_tid += 1

        click_x, click_y = self._pending_click_full
        contour_rel = self._pending_contour_rel.reshape(-1, 2).astype(int).tolist()

        t = TargetSpec(
            tid=tid,
            click_x=int(click_x),
            click_y=int(click_y),
            init_contour_xy=contour_rel,
            box_mode=bool(self._pending_box_mode),
        )
        self.targets_full.append(t)

        mode_str = "BOX" if bool(self._pending_box_mode) else "CONTOUR"
        item = QListWidgetItem(f"Target {tid}: {mode_str} click=({click_x},{click_y}) | pts={len(contour_rel)}")
        self.targets_list.addItem(item)

        self.mode = "idle"
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self._reset_preview_pixmap()

        self.status_label.setText(f"Added target {tid}. Define Target to add another, or run Quick Test / RUN.")
        self._sync_buttons()

    def on_remove_last(self):
        if not self.targets_full:
            return
        removed = self.targets_full.pop()
        self.targets_list.takeItem(self.targets_list.count() - 1)
        self.status_label.setText(f"Removed target {removed.tid}.")
        self._sync_buttons()

    def on_clear(self):
        self.preview.clearRect()
        self.mode = "idle"
        self.arena_full = None
        self._pending_click_full = None
        self._pending_contour_rel = None
        self._pending_box_mode = False
        self.targets_full = []
        self.next_tid = 1
        self.targets_list.clear()
        self._reset_preview_pixmap()
        self.status_label.setText("Cleared. Define Arena → draw → Save Arena (or Use Full Frame Arena).")
        self._sync_buttons()

    def _build_config(self) -> SessionConfig:
        if self.video_path is None or self.first_frame_bgr is None:
            raise RuntimeError("No video loaded.")
        if self.arena_full is None:
            raise RuntimeError("Arena not set. Define Arena and Save Arena first (or Use Full Frame Arena).")
        if not self.targets_full:
            raise RuntimeError("No targets added. Define Target then Add Target at least once.")

        session = self.session_name.text().strip() or self.video_path.stem
        out_root = Path(self.output_root.text().strip() or (Path.home() / "AntHeatmaps"))
        out_dir = out_root / session

        ax, ay, aw, ah = self.arena_full

        return SessionConfig(
            created_at=now_iso(),
            session_name=session,
            video_path=str(self.video_path),
            output_dir=str(out_dir),
            arena_x=int(ax),
            arena_y=int(ay),
            arena_w=int(aw),
            arena_h=int(ah),
            targets=list(self.targets_full),

            tiny_mode=bool(self.tiny_mode.isChecked()),
            processing_scale=float(self.processing_scale.value()),
            search_radius=int(self.search_radius.value()),
            lost_patience_frames=int(self.lost_patience.value()),

            diff_thresh=int(self.diff_thresh.value()),
            bg_alpha=float(self.bg_alpha.value()),
            min_blob_area=int(self.min_area.value()),
            seg_blur_sigma=float(self.seg_blur.value()),

            heat_sigma=float(self.heat_sigma.value()),
            heat_post_blur=float(self.heat_post_blur.value()),
            clip_percentile=float(self.clip_pct.value()),
            gamma=float(self.gamma.value()),
            overlay_alpha=float(self.alpha.value()),
            export_overlay_video=bool(self.export_overlay.isChecked()),
            overlay_video_fps=float(self.overlay_fps.value()),
            show_live_tracking=bool(self.show_live_tracking.isChecked()),
            warmup_frames=25,
        )

    def on_quick_run(self):
        self._start_run(float(self.quick_seconds.value()))

    def on_full_run(self):
        self._start_run(0.0)

    def _start_run(self, quick_seconds: float):
        try:
            cfg = self._build_config()
        except Exception as e:
            QMessageBox.warning(self, "Config error", str(e))
            return

        out_dir = Path(cfg.output_dir)
        in_dir = out_dir / "input"
        safe_mkdir(in_dir)

        try:
            dst = in_dir / Path(cfg.video_path).name
            if Path(cfg.video_path).resolve() != dst.resolve():
                if not dst.exists():
                    import shutil
                    shutil.copy2(cfg.video_path, dst)
                cfg.video_path = str(dst)
        except Exception:
            pass

        self._lock_ui(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting...")
        self.preview.clearRect()

        self.worker = ProcessWorker(cfg, quick_test_seconds=quick_seconds)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        if cfg.show_live_tracking:
            self.worker.live_frame.connect(self._on_live_frame)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_live_frame(self, qimg: QImage):
        self.preview.setPixmap(QPixmap.fromImage(qimg))

    def _on_finished_ok(self, out_folder: str):
        self._lock_ui(False)
        self.worker = None
        QMessageBox.information(
            self,
            "Done",
            "Finished!\n\n"
            f"Outputs saved to:\n{out_folder}\n\n"
            "Check outputs:\n"
            "overlay_multi.mp4\n"
            "heatmap_target_###_overlay.png\n"
            "heatmap_combined_overlay.png\n"
            "underexplored_map.png\n"
            "underexplored_stats.csv\n"
            "tracks_multi.csv"
        )
        try:
            if sys.platform == "darwin":
                os.system(f'open "{out_folder}"')
        except Exception:
            pass

    def _on_failed(self, msg: str):
        self._lock_ui(False)
        self.worker = None
        QMessageBox.critical(self, "Run failed", msg)

    def _lock_ui(self, running: bool):
        for w in [
            self.btn_import, self.btn_define_arena, self.btn_full_arena, self.btn_save_arena,
            self.btn_define_target, self.btn_add_target, self.btn_remove_last, self.btn_clear,
            self.session_name, self.output_root, self.overlay_fps, self.quick_seconds,
            self.export_overlay, self.show_live_tracking, self.search_radius, self.lost_patience,
            self.tiny_mode, self.processing_scale,
            self.diff_thresh, self.bg_alpha, self.min_area, self.seg_blur,
            self.heat_sigma, self.heat_post_blur, self.clip_pct, self.gamma, self.alpha,
            self.btn_quick, self.btn_run
        ]:
            w.setEnabled(not running)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
