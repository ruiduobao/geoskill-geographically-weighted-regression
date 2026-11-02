# Geographically Weighted Regression (geoskill-geographically-weighted-regression)

> GWR local regression + bandwidth selection + local coefficient maps + local R2 (Fotheringham/Brunsdon/Charlton paradigm, bisquare/gaussian kernels, AICc bandwidth selection)

---

## 1. Overview

Geographically Weighted Regression (GWR): at each regression point, a distance-decay kernel (bisquare/Gaussian) is used for a weighted least squares fit, yielding local coefficients that vary over space; the optimal bandwidth is selected via **AICc cross-validation** (Hurvich & Tsai 1989); outputs local coefficient rasters and local R².

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-geographically-weighted-regression.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `local_coefficients.tif` | GeoTIFF (k bands) | Band 1 = intercept, bands 2+ = local coefficient of each independent variable |
| `local_r2.tif` | GeoTIFF (single band) | Local R² raster (IDW interpolated from sample points) |
| `gwr_stats.json` | JSON | Optimal bandwidth, AICc, tr(S), R², bandwidth search curve, mean coefficients |
| `output-manifest.json` | JSON | Run manifest |

## 6. Technical Principle

- **Kernel functions**:
  - Bisquare (bisquare / Tukey biweight): `w(d,h) = (1 - (d/h)²)² for d<h, 0 otherwise`
  - Gaussian (gaussian): `w(d,h) = exp(-0.5·(d/h)²)`
- **Point-wise weighted OLS**: `β_i = (X^T W_i X)^{-1} X^T W_i y` (Fotheringham et al. 2002, Chapter 4)
- **Bandwidth selection**: six fixed bandwidth candidates (0.1× / 0.2× / 0.3× / 0.5× / 0.7× / 1.0× of the data span), with **AICc** (Hurvich & Tsai 1989 correction) selecting the optimum
- **AICc formula**: `AICc = n·log(σ²) + n·log(2π) + n·(n + tr(S))/(n − 2 − tr(S))`, where `tr(S) = Σ_i x_i^T (X^T W_i X)^{-1} x_i·w_i` is the trace of the hat matrix
- **Local R²**: at each regression point, `1 − resid²/var(y)` clipped to [0,1] (per-point residual normalization, used as a proxy for local goodness of fit)

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
