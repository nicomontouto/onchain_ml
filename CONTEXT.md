# Onchain ML — Contexto del Proyecto

## Estado actual
- **Versión:** v5 (en producción / subida a GitHub)
- **Repo:** github.com/nicomontouto/onchain_ml
- **Punto de entrada:** `python main_v5.py`

---

## Decisiones clave tomadas

### Por qué solo 4h
La frecuencia daily fue descartada por falta de datos (~1,825 barras, insuficiente para ML robusto con walk-forward). Solo se usa 4h con ~10,000 barras.

### Por qué no LSTM/GRU
Exploradas en versiones anteriores. Descartadas porque el volumen de datos es insuficiente para entrenar redes recurrentes sin overfitting. Se usa MLP (16×3) como única red neuronal.

### Data leakage v4 → v5
v4 usaba datos diarios (whale_balance_pct, herfindahl_index, holders_growth, fear_greed_value) asignados a cada barra de 4h del mismo día. Eso le daba al modelo acceso implícito a información del futuro. Solución: reemplazar por fuentes 100% horarias (The Graph 1h, Dune 1h).

### Reducción de features (180 → 10)
Proceso en 3 etapas:
1. Eliminar features con granularidad diaria (leakage)
2. Calcular MI + Distance Correlation + Spearman vs label
3. Filtro de correlación de Pearson > 0.85 entre features

Descartadas explícitamente:
- `log_return` — MI=0, dCor bajo
- `volume_tvl_ratio` — Spearman=0.997 con feesUSD
- `feesUSD` — Spearman=0.972 con volumeUSD_roll_std72
- `transfer_count_pct_change` — MI~0
- `price_divergence_lag1` — redundante con price_divergence

### Dollar bars
Umbral $6.9M → ~10,000 barras completadas, promedio 4.4h/barra. La feature `dollar_bar_duration` captura la intensidad del flujo de capital. Se construyó con propagación de exceso (while loop) para manejar horas con volumen > umbral.

---

## Triple Barrier Method (parámetros actuales)

| Parámetro | Valor | Razonamiento |
|-----------|-------|--------------|
| `ptSl` | [2.0, 2.0] | Multiplier 2× vol para generar ~25% neutrales |
| `t1` | 12 barras | 48h horizonte |
| `vol_span` | 20 barras | Rolling std con shift(1) para evitar lookahead |
| Filtro | CUSUM simétrico | h = trgt.mean(), reduce 10k → 3,747 eventos |
| `side` | None (=1) | Sin prior direccional — label=+1 siempre significa precio subió |

**Distribución de labels:** +1=38%, 0=25%, -1=37%

### Por qué side=None
Se intentó `side = np.sign(close.pct_change(4).shift(1))` para meta-labeling lite. Problema: con side variable, label=+1 significa "momentum continuó" (no necesariamente precio subió). La simulación siempre interpreta P(+1) como señal LONG → mismatch → pérdidas sistemáticas. Solución: volver a side=None.

### Volatilidad con shift
`vol = ret.shift(1).rolling(20).std()` — el shift(1) es importante para que vol[t] no use el retorno del bar actual (lookahead leve).

---

## Features finales (10)

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `log_return_lag1` | Binance | Log-retorno barra anterior |
| `log_return_lag4` | Binance | Log-retorno 4 barras atrás (16h) |
| `volatility_24h` | Binance | Rolling std 6 barras (24h) |
| `high_low_range` | Binance | (high-low)/close |
| `log_return_roll_std72` | Binance | Rolling std 18 barras (72h) |
| `price_divergence` | Binance+TheGraph | \|CEX - DEX\| / DEX |
| `volumeUSD_roll_std72` | The Graph | Std volumen DEX 18 barras |
| `transfer_count` | Dune | Transferencias UNI en 4h |
| `whale_volume_ratio` | Dune | whale_vol / total_vol |
| `dollar_bar_duration` | Binance | Duración última barra $6.9M |

---

## Resultados (test set, último run)

Split 80/20 temporal. Capital $10,000. Trade size $500. Fee 0.1%.

| Modelo | Net P&L | Sharpe | Max DD | Acc | F1-macro | Trades | %Flat |
|--------|---------|--------|--------|-----|----------|--------|-------|
| XGB-4h | +$75.82 | 0.330 | -4.60% | 0.431 | 0.411 | 89 | 44% |
| LGBM-4h | +$21.60 | 0.112 | -4.84% | 0.424 | 0.395 | 94 | 43% |
| MLP-4h | -$137.84 | -0.538 | -4.97% | 0.421 | 0.409 | 110 | 44% |
| BH-4h | -$3,408 | 0.167 | -74.82% | — | — | 1 | 0% |

El período de test coincidió con mercado bajista severo para UNI. Los 3 modelos superan ampliamente al BH en drawdown (-5% vs -75%).

**Walk-forward validation (XGB, 5 folds sobre train):**
- Folds 1-3: acc ~0.37-0.40, f1m ~0.33
- Folds 4-5: acc ~0.42-0.44, f1m ~0.42 (mejora con más datos)

---

## Fuentes de datos

| Fuente | Archivo | Granularidad | Rango |
|--------|---------|-------------|-------|
| Binance API | `data/cache/binance_uniusdt_1h_1825d.parquet` | 1h | 2021-04 → 2026-04 |
| The Graph | `data/thegraph_uni_hourly.csv` | 1h | 2021-05 → 2026-04 |
| Dune Analytics | `data/dune_uni_activity_1h.csv` | 1h | 2020-09 → 2026-04 |
| Dollar bars | `data/dollar_bar_duration.csv` | Variable | 2021-04 → 2026-04 |

**API keys** en `.env` (gitignored):
- `THEGRAPH_API_KEY`
- `DUNE_API_KEY`
- `ETHERSCAN_API_KEY`

---

## Archivos del pipeline

```
main_v5.py              Orquestador
process_data_v5.py      Carga + merge horario + resample 4h + features
feature_selection_v5.py Selección de features
triple_barrier_v4.py    Labeling TBM + CUSUM
models_v4.py            XGBWrapper, LGBMWrapper, MLPWrapper
simulate_v4.py          Simulación LONG/SHORT/FLAT + Buy & Hold
generate_report_v5.py   PDF (gitignored, no subido)
```

---

## Cosas que quedaron pendientes / posibles mejoras

- El dataset de test es pequeño (750 eventos). Con más datos históricos se podría validar en distintos regímenes de mercado.
- MLP ligeramente negativo — consistente con menos poder predictivo que tree-based.
- LGBM early stopping en iter 47 (<50) — podría beneficiarse de más iteraciones o ajuste de learning rate.
- No se implementó sample uniqueness / fractional differentiation (LdP cap. 4) para reducir correlación serial entre eventos solapados.
- El proyecto es académico, no pensado para producción.
