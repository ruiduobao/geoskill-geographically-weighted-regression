#!/usr/bin/env python3
"""geographically-weighted-regression — 地理加权回归 (GWR)

对空间样本拟合地理加权回归（GWR）。在每个回归点用距离衰减权重做加权最小
二乘，得到随空间变化的局部系数；通过交叉验证选择最优带宽；输出局部系数栅格
与局部 R²。

数据源：本地 CSV，或 --synthetic 生成系数随空间变化的模拟样本。

隐私声明 / Privacy：
- 默认离线运行，不发起任何网络请求。
- 所有处理在本地完成，不上传用户数据。

Usage:
    python geographically-weighted-regression.py --bbox 116 39 117 40 --synthetic --output-dir ./out

License: MIT
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SKILL_NAME = "geographically-weighted-regression"

try:
    from _geoskill_core.errors import (
        GeoSkillError, UsageError, ValidationError, ProcessError, to_exit_code,
    )
    from _geoskill_core.manifest import OutputManifest, OutputFile
except ImportError:  # pragma: no cover
    class GeoSkillError(Exception):
        def __init__(self, message: str, code: int = 7, kind: str = "EGeo", **kw):
            super().__init__(message)
            self.message, self.code, self.kind = message, code, kind

    class UsageError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=2, kind="EUsage", **k)

    class ValidationError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=6, kind="EValidate", **k)

    class ProcessError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=7, kind="EProcess", **k)

    def to_exit_code(exc):
        return getattr(exc, "code", 7)

    OutputManifest = None
    OutputFile = None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def validate_bbox(bbox) -> List[float]:
    """校验 bbox [W, S, E, N]；不合法抛 ValidationError（exit 6）。

    同 buffer-analysis / change-detection-dl：跨 180°（W > E）静默产出负
    像元宽度错查询；超经纬度范围亦然。统一前置校验，给出可读提示。
    """
    if bbox is None:
        return None  # type: ignore[return-value]
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"bbox values must be numeric: {bbox}") from exc
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError(f"bbox longitude out of range [-180, 180]: W={w}, E={e}")
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError(f"bbox latitude out of range [-90, 90]: S={s}, N={n}")
    if w > e:
        raise ValidationError(
            f"bbox W ({w}) > E ({e}); antimeridian-crossing bbox is not supported — "
            "split the request into two bboxes on either side of +/-180")
    if s > n:
        raise ValidationError(f"bbox S ({s}) > N ({n})")
    return [w, s, e, n]


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------
def bisquare_kernel(d: np.ndarray, h: float) -> np.ndarray:
    """双平方核权重 w = (1 - (d/h)^2)^2 (d<h)，否则 0。"""
    u = d / h
    w = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
    return w


def gaussian_kernel(d: np.ndarray, h: float) -> np.ndarray:
    """高斯核权重 w = exp(-0.5 (d/h)^2)。"""
    return np.exp(-0.5 * (d / h) ** 2)


def gwr_fit(
    coords: np.ndarray, X: np.ndarray, y: np.ndarray, bandwidth: float,
    kernel: str = "bisquare",
) -> Dict[str, Any]:
    """逐点 GWR 加权最小二乘。

    Parameters
    ----------
    coords : (n, 2)
    X : (n, k) 含截距
    y : (n,)
    bandwidth : 核带宽（距离单位）

    Returns dict: local_beta (n,k), yhat (n,), resid (n,), local_r2 (n,),
                  tr_hat (帽矩阵迹), aicc。
    """
    n, k = X.shape
    kfunc = bisquare_kernel if kernel == "bisquare" else gaussian_kernel
    local_beta = np.zeros((n, k))
    yhat = np.zeros(n)
    hat_diag = np.zeros(n)  # 帽矩阵对角（用于自由度）
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(1))
        w = kfunc(d, bandwidth)
        Wsqrt = np.sqrt(w)
        Xw = X * Wsqrt[:, None]
        yw = y * Wsqrt
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(k)
        local_beta[i] = beta
        yhat[i] = X[i] @ beta
        # 帽矩阵对角元素 h_ii = x_i (X'WX)^-1 X'W 的第 i 列
        try:
            XtWX = Xw.T @ Xw
            XtWX_inv = np.linalg.inv(XtWX + 1e-10 * np.eye(k))
            ri = (X[i] * w[i])  # x_i * w_i
            hat_diag[i] = X[i] @ XtWX_inv @ ri
        except np.linalg.LinAlgError:
            hat_diag[i] = 1.0
    resid = y - yhat
    rss = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    tr_hat = float(hat_diag.sum())
    # AICc
    sigma2 = rss / n
    aicc = np.inf
    if sigma2 > 0 and n - 2 - tr_hat > 0:
        aicc = n * np.log(sigma2) + n * np.log(2 * np.pi) + n * (
            (n + tr_hat) / (n - 2 - tr_hat))
    # 局部 R²：逐点残差方差 vs 全局方差
    local_r2 = np.clip(1.0 - resid ** 2 / max(sst / n, 1e-12), 0.0, 1.0)
    r2_global = 1.0 - rss / sst if sst > 0 else 0.0
    return {"local_beta": local_beta, "yhat": yhat, "resid": resid,
            "local_r2": local_r2, "tr_hat": tr_hat, "aicc": float(aicc),
            "r2": float(r2_global)}


def select_bandwidth(
    coords: np.ndarray, X: np.ndarray, y: np.ndarray,
    candidates: np.ndarray, kernel: str = "bisquare",
) -> Tuple[float, List[Dict[str, Any]]]:
    """用 AICc 在候选带宽中选最优。返回 (best_bw, 各候选记录)。"""
    records = []
    best = {"bw": float(candidates[0]), "aicc": np.inf}
    for bw in candidates:
        res = gwr_fit(coords, X, y, float(bw), kernel)
        rec = {"bandwidth": float(bw), "aicc": res["aicc"],
               "r2": res["r2"], "tr_hat": res["tr_hat"]}
        records.append(rec)
        if res["aicc"] < best["aicc"]:
            best = {"bw": float(bw), "aicc": res["aicc"]}
    return best["bw"], records


# ---------------------------------------------------------------------------
# 合成数据：系数随空间变化
# ---------------------------------------------------------------------------
def generate_synthetic(bbox: List[float], n_points: int = 150, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """生成 GWR 样本：x1 的系数随经度线性变化。

    Returns (coords, X, y, beta1_true_field, info)。
    """
    rng = np.random.default_rng(seed)
    w, s, e, n = bbox
    coords = rng.uniform([w, s], [e, n], (n_points, 2))
    x1 = rng.normal(0, 1, n_points)
    X = np.column_stack([np.ones(n_points), x1])
    # 截距 = 1；x1 系数随归一化经度从 0.5 变到 3.0
    u = (coords[:, 0] - w) / max(e - w, 1e-9)
    beta1_true = 0.5 + 2.5 * u
    beta0_true = np.full(n_points, 1.0)
    y = beta0_true + beta1_true * x1 + rng.normal(0, 0.3, n_points)
    info = {"n_points": n_points, "beta1_range": [float(beta1_true.min()), float(beta1_true.max())]}
    return coords, X, y, beta1_true, info


# ---------------------------------------------------------------------------
# GeoTIFF I/O / Manifest
# ---------------------------------------------------------------------------
def write_geotiff(path: str, array: np.ndarray, bbox: List[float], nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValidationError("write_geotiff expects a 2D or 3D array")
    nb, hh, ww = array.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], ww, hh)
    profile = {
        "driver": "GTiff", "height": hh, "width": ww, "count": nb,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nb):
            dst.write(array[b].astype("float32"), b + 1)


def idw_to_grid(coords: np.ndarray, values: np.ndarray, grid_x: np.ndarray,
                grid_y: np.ndarray, power: float = 2.0) -> np.ndarray:
    """简单 IDW 把点值插到规则网格（用于系数空间图）。"""
    h, w = grid_x.shape
    gx = grid_x.ravel()
    gy = grid_y.ravel()
    out = np.zeros(gx.shape[0])
    for i in range(gx.shape[0]):
        d = np.sqrt((coords[:, 0] - gx[i]) ** 2 + (coords[:, 1] - gy[i]) ** 2)
        with np.errstate(divide="ignore"):
            wt = np.where(d > 1e-12, 1.0 / d ** power, 0.0)
        exact = d < 1e-12
        if exact.any():
            out[i] = values[exact][0]
        else:
            out[i] = (wt * values).sum() / wt.sum()
    return out.reshape(h, w)


def write_manifest(
    output_dir: str, args: argparse.Namespace, outputs: List[Dict[str, Any]],
    qa: Dict[str, Any], started_at: str, exit_code: int, bbox: List[float],
) -> Optional[str]:
    if OutputManifest is None:
        return None
    cmd = " ".join([SKILL_NAME] + sys.argv[1:])
    man = OutputManifest(
        skill=SKILL_NAME, skill_version=VERSION, command=cmd,
        started_at=started_at, finished_at=_utc_now(), exit_code=exit_code,
        inputs={"input": getattr(args, "input", None), "synthetic": bool(getattr(args, "synthetic", False))},
        outputs=[OutputFile(**o) for o in outputs],
        qa=qa,
        software={"python": sys.version.split()[0], "skill": SKILL_NAME},
    )
    path = os.path.join(output_dir, "output-manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bbox = list(args.bbox) if args.bbox else None

    if args.input and not args.synthetic:
        if not os.path.exists(args.input):
            raise UsageError(f"input not found: {args.input}", path=args.input)
        coords, X, y = _load_csv(args.input)
        if bbox is None:
            bbox = [float(coords[:, 0].min()), float(coords[:, 1].min()),
                    float(coords[:, 0].max()), float(coords[:, 1].max())]
        bbox = validate_bbox(bbox)
        source_note = args.input
    else:
        if bbox is None:
            raise UsageError("provide --bbox (synthetic mode) or --input <csv>")
        bbox = validate_bbox(bbox)
        if not isinstance(args.n_points, int) or args.n_points <= 0:
            raise ValidationError(
                f"--n-points must be a positive integer; got {args.n_points!r}")
        coords, X, y, _, _ = generate_synthetic(bbox, n_points=args.n_points)
        source_note = "synthetic"

    if coords.shape[0] < X.shape[1] + 5:
        raise ValidationError("too few observations for GWR")

    # 候选带宽：基于坐标跨度的分数
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1e-6)
    fracs = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    candidates = fracs * span
    best_bw, bw_records = select_bandwidth(coords, X, y, candidates, args.kernel)
    res = gwr_fit(coords, X, y, best_bw, args.kernel)

    # 局部系数栅格化
    gs = args.grid_size
    gx = np.linspace(bbox[0], bbox[2], gs)
    gy = np.linspace(bbox[3], bbox[1], gs)
    gxx, gyy = np.meshgrid(gx, gy)
    k = X.shape[1]
    coef_grids = []
    for j in range(k):
        coef_grids.append(idw_to_grid(coords, res["local_beta"][:, j], gxx, gyy))
    coef_cube = np.stack(coef_grids, axis=0).astype(np.float32)
    r2_grid = idw_to_grid(coords, res["local_r2"], gxx, gyy).astype(np.float32)

    out_coef = os.path.join(output_dir, "local_coefficients.tif")
    write_geotiff(out_coef, coef_cube, bbox)
    out_r2 = os.path.join(output_dir, "local_r2.tif")
    write_geotiff(out_r2, r2_grid, bbox)
    stats_path = os.path.join(output_dir, "gwr_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"best_bandwidth": best_bw, "kernel": args.kernel,
                   "bandwidth_search": bw_records, "aicc": res["aicc"],
                   "r2": res["r2"], "tr_hat": res["tr_hat"],
                   "coef_mean_per_var": [float(c.mean()) for c in coef_grids]},
                  f, ensure_ascii=False, indent=2)

    qa = {
        "source": source_note,
        "n_obs": int(coords.shape[0]),
        "best_bandwidth": float(best_bw),
        "aicc": res["aicc"],
        "r2": res["r2"],
    }
    outputs = [
        {"path": out_coef, "kind": "raster", "crs_epsg": 4326, "bbox_wgs84": bbox, "band_count": int(k)},
        {"path": out_r2, "kind": "raster", "crs_epsg": 4326, "bbox_wgs84": bbox, "band_count": 1},
        {"path": stats_path, "kind": "json"},
    ]
    man_path = write_manifest(output_dir, args, outputs, qa, started_at, 0, bbox)

    if not args.quiet:
        print(f"[{SKILL_NAME}] source: {source_note}  n={coords.shape[0]}")
        print(f"[{SKILL_NAME}] best bandwidth: {best_bw:.4f} ({args.kernel})")
        print(f"[{SKILL_NAME}] GWR R²={res['r2']:.4f}  AICc={res['aicc']:.2f}")
        print(f"[{SKILL_NAME}] coef means: {[round(float(c.mean()),3) for c in coef_grids]}")
        print(f"[{SKILL_NAME}] output: {out_coef}")
        if man_path:
            print(f"[{SKILL_NAME}] manifest: {man_path}")
    return 0


def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValidationError("CSV has no data rows")
    header = [h.strip().lower() for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=np.float64)
    col = {h: i for i, h in enumerate(header)}
    if "x" not in col or "y" not in col or "dep" not in col:
        raise ValidationError("CSV must contain columns: x, y, dep")
    coords = data[:, [col["x"], col["y"]]]
    y = data[:, col["dep"]]
    ind_cols = [i for h, i in col.items() if h.startswith("ind")]
    inds = data[:, ind_cols] if ind_cols else np.zeros((data.shape[0], 0))
    X = np.column_stack([np.ones(data.shape[0]), inds])
    return coords, X, y


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SKILL_NAME,
        description="Geographically Weighted Regression: local coefficients, bandwidth selection, local R².",
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="geographic extent [minLon minLat maxLon maxLat]")
    p.add_argument("--input", help="input CSV (x,y,dep,ind...)")
    p.add_argument("--n-points", type=int, default=150, help="synthetic sample size (default: 150)")
    p.add_argument("--kernel", default="bisquare", choices=["bisquare", "gaussian"],
                   help="kernel function (default: bisquare)")
    p.add_argument("--grid-size", type=int, default=32, help="output coefficient grid size (default: 32)")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data")
    p.add_argument("--output-dir", default="./output", help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress output")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return process(args)
    except GeoSkillError as exc:
        print(f"[{SKILL_NAME}] ERROR [{exc.kind}] {exc.message}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[{SKILL_NAME}] ERROR {exc}", file=sys.stderr)
        return to_exit_code(exc)


if __name__ == "__main__":
    sys.exit(main())
