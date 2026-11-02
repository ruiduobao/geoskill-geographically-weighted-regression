---
name: geoskill-geographically-weighted-regression
description: 'GWR局部回归+带宽选择+局部系数空间图+局部R2 (Fotheringham/Brunsdon/Charlton 范式, bisquare/gaussian核, AICc带宽选择)'
---

# 地理加权回归 | Geographically Weighted Regression

Geographically weighted regression (GWR): at each regression point, a distance-decay kernel (bisquare/gaussian) is used for weighted least squares, producing local coefficients that vary in space; the optimal bandwidth is selected by **AICc cross-validation** (Hurvich & Tsai 1989); outputs include local coefficient rasters and local R².

## 核心算法（Fotheringham/Brunsdon/Charlton 范式） / 核心算法（Fotheringham/Brunsdon/Charlton 范式）

- **Kernel functions**:
  - Bisquare (Tukey biweight): `w(d,h) = (1 - (d/h)²)² for d<h, 0 otherwise`
  - Gaussian: `w(d,h) = exp(-0.5·(d/h)²)`
- **Pointwise weighted OLS**: `β_i = (X^T W_i X)^{-1} X^T W_i y` (Fotheringham et al. 2002, Chapter 4)
- **Bandwidth selection**: 6 fixed bandwidth candidates (0.1× / 0.2× / 0.3× / 0.5× / 0.7× / 1.0× of the data extent), with the best chosen by **AICc** (Hurvich & Tsai 1989 correction)
- **AICc formula**: `AICc = n·log(σ²) + n·log(2π) + n·(n + tr(S))/(n − 2 − tr(S))`, where `tr(S) = Σ_i x_i^T (X^T W_i X)^{-1} x_i·w_i` is the trace of the hat matrix
- **Local R²**: `1 − resid²/var(y)` at each regression point, clipped to [0,1] (per-point residual normalization, used as a proxy for local goodness of fit)

## Methodology References / 方法学引用

| Source | Purpose |
|---|---|
| Fotheringham, Brunsdon, Charlton, *Geographically Weighted Regression*, Wiley 2002 | GWR paradigm, kernel functions, AICc, local coefficients, R² |
| Brunsdon, Fotheringham, Charlton, *Geographical Analysis* 28(4):281-298, 1996, DOI:10.1111/j.1538-4632.1996.tb00936.x | Original GWR paper |
| Hurvich, Tsai, *Biometrika* 76(2):297-307, 1989, DOI:10.1093/biomet/76.2.297 | AICc formula |
| Páez, Farber, Wheeler, *Environment and Planning B* 38(6):1075-1098, 2011, DOI:10.1068/b100708j | Systematic comparison of bisquare vs gaussian / fixed vs adaptive bandwidths |

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy
```

## Usage / 使用方法

### Example 1 (synthetic data, offline)

```bash
python geoskill-geographically-weighted-regression.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2 (specify area + silence)

```bash
python geoskill-geographically-weighted-regression.py --bbox 121.0 31.0 122.0 32.0 --synthetic --output-dir ./out2 --quiet
```

### Example 3 (real CSV input)

```bash
python geoskill-geographically-weighted-regression.py --input housing.csv --output-dir ./out3
```

CSV columns: must contain `x, y, dep`; optionally multiple independent variables `ind1, ind2, ...` (no intercept; the script adds it automatically).

### Example 4 (tiny-area boundary test)

```bash
python geoskill-geographically-weighted-regression.py --bbox 116.39 39.90 116.40 39.91 --synthetic --output-dir ./out4 --quiet
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `local_coefficients.tif` | GeoTIFF (k bands) | Band 1 = intercept; bands 2+ = local coefficient per independent variable |
| `local_r2.tif` | GeoTIFF (single band) | Local R² raster (IDW-interpolated from sample points) |
| `gwr_stats.json` | JSON | Optimal bandwidth, AICc, tr(S), R², bandwidth search curve, coefficient means |
| `output-manifest.json` | JSON | Run manifest |

## Limitations / 局限

- **Distance metric**: Euclidean distance in degree coordinate space. The maximum error within 1°×1° is <1% (cos 39° ≈ 0.777), but the bias grows over large areas / at high latitudes. Fotheringham recommends great-circle distance (haversine), which is not implemented.
- **Local R²**: per-point residual normalization (pseudo R²), not Fotheringham's standard "weighted RSS/TSS within the bandwidth". Both are widely used but have different meanings.
- **Significance tests**: local coefficient t-tests / pseudo t-tests (Fotheringham Chapter 6) are not implemented.
- **Fixed bandwidth**: distance-based only; adaptive (neighbor-count) bandwidths are not implemented.
- **CSV NoData/NaN**: `_load_csv` uses `float(v)`; a "NaN" string raises ValueError (exit 2); no NoData filtering.

## Data Source / 数据源 / Source

- Synthetic mode: locally generated simulated samples with spatially varying coefficients.
- Real mode: reads a local CSV (`x, y, dep, ind...`), no network requests.

## Privacy / 隐私声明 / Privacy

- Fully offline by default; no network requests are made.
- `--synthetic` mode reads no external data.
- All computation is performed locally; user data is never uploaded.

## License / License

MIT

---

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
