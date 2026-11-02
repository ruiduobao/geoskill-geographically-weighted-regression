# 地理加权回归 | Geographically Weighted Regression

地理加权回归（GWR）：在每个回归点用距离衰减核（双平方/高斯）做加权最小二乘，得到随空间变化的局部系数；用 **AICc 交叉验证**（Hurvich & Tsai 1989）选最优带宽；输出局部系数栅格与局部 R²。

## 核心算法（Fotheringham/Brunsdon/Charlton 范式）

- **核函数**：
  - 双平方（bisquare / Tukey biweight）：`w(d,h) = (1 - (d/h)²)² for d<h, 0 otherwise`
  - 高斯（gaussian）：`w(d,h) = exp(-0.5·(d/h)²)`
- **逐点加权 OLS**：`β_i = (X^T W_i X)^{-1} X^T W_i y`（Fotheringham et al. 2002, Chapter 4）
- **带宽选择**：6 个固定带宽候选（数据跨度的 0.1× / 0.2× / 0.3× / 0.5× / 0.7× / 1.0×），用 **AICc**（Hurvich & Tsai 1989 修正）选最优
- **AICc 公式**：`AICc = n·log(σ²) + n·log(2π) + n·(n + tr(S))/(n − 2 − tr(S))`，其中 `tr(S) = Σ_i x_i^T (X^T W_i X)^{-1} x_i·w_i` 是帽矩阵迹
- **局部 R²**：每个回归点的 `1 − resid²/var(y)` 截到 [0,1]（per-point 残差归一化，作为局部拟合优度的代理指标）

## 方法学引用

| 来源 | 用途 |
|---|---|
| Fotheringham, Brunsdon, Charlton, *Geographically Weighted Regression*, Wiley 2002 | GWR 范式、核函数、AICc、局部系数、R² |
| Brunsdon, Fotheringham, Charlton, *Geographical Analysis* 28(4):281-298, 1996, DOI:10.1111/j.1538-4632.1996.tb00936.x | GWR 原始论文 |
| Hurvich, Tsai, *Biometrika* 76(2):297-307, 1989, DOI:10.1093/biomet/76.2.297 | AICc 公式 |
| Páez, Farber, Wheeler, *Environment and Planning B* 38(6):1075-1098, 2011, DOI:10.1068/b100708j | bisquare vs gaussian / 固定 vs adaptive 带宽系统对比 |

## 依赖

```bash
pip install numpy rasterio scipy
```

## 使用方法

### 示例 1（合成数据，离线）

```bash
python geoskill-geographically-weighted-regression.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### 示例 2（指定区域 + 静默）

```bash
python geoskill-geographically-weighted-regression.py --bbox 121.0 31.0 122.0 32.0 --synthetic --output-dir ./out2 --quiet
```

### 示例 3（真实输入 CSV）

```bash
python geoskill-geographically-weighted-regression.py --input housing.csv --output-dir ./out3
```

CSV 列：必须含 `x, y, dep`；可选多个 `ind1, ind2, ...` 自变量（不含截距，脚本自动加）。

### 示例 4（极小区域边界测试）

```bash
python geoskill-geographically-weighted-regression.py --bbox 116.39 39.90 116.40 39.91 --synthetic --output-dir ./out4 --quiet
```

## 输出

| 文件 | 格式 | 说明 |
|---|---|---|
| `local_coefficients.tif` | GeoTIFF (k 波段) | 第 1 波段=截距，第 2+ 波段=各自变量局部系数 |
| `local_r2.tif` | GeoTIFF (单波段) | 局部 R² 栅格（IDW 插值自样本点） |
| `gwr_stats.json` | JSON | 最优带宽、AICc、tr(S)、R²、带宽搜索曲线、系数均值 |
| `output-manifest.json` | JSON | 运行清单 |

## 局限

- **距离度量**：欧氏距离在度坐标空间。1°×1° 内最大误差 <1%（cos 39°≈0.777），大区域/高纬度偏差增大。Fotheringham 推荐用大圆距离（haversine），未实现。
- **局部 R²**：per-point 残差归一化（pseudo R²），非 Fotheringham 标准的"带宽内加权 RSS/TSS"。两者皆被广泛使用，含义不同。
- **显著性检验**：局部系数 t 检验 / 伪 t 检验（Fotheringham Chapter 6）未实现。
- **固定带宽**：仅 distance-based，未实现 adaptive（neighbor-count）带宽。
- **CSV NoData/NaN**：`_load_csv` 用 `float(v)`，遇 "NaN" 字符串抛 ValueError（exit 2），未做 NoData 过滤。

## 数据源 / Source

- 合成模式：本地生成系数随空间变化的模拟样本。
- 真实模式：读取本地 CSV（`x, y, dep, ind...`），无网络请求。

## 隐私声明 / Privacy

- 默认完全离线运行，不发起任何网络请求。
- `--synthetic` 模式不读取任何外部数据。
- 所有计算在本地完成，不上传用户数据。

## License

MIT
