# Onchain ML — Pipeline de Trading Algorítmico con Datos On-Chain (UNI/USDT)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189fdd)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-2ecc71)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e?logo=scikit-learn)

## Descripción

Pipeline end-to-end de machine learning para construir y backtestear estrategias de trading algorítmico sobre el par UNI/USDT. El proyecto combina métricas **on-chain del protocolo Uniswap** con **datos de precio de Binance** y **actividad de transferencias on-chain** para entrenar clasificadores que predicen la dirección del precio en el siguiente período de 4 horas.

El pipeline genera señales LONG/SHORT/FLAT y simula una cuenta de capital real ($10,000), comparando el desempeño contra un benchmark Buy & Hold con capital completo expuesto.

Este proyecto es de carácter académico/exploratorio, no está pensado para producción.

---

## Actualizaciones v6

En esta versión se incorporaron tres cambios principales que transformaron el rendimiento del pipeline:

**Sequential Feature Selection (SFS).** Se reemplazó el proceso manual de selección de features por un algoritmo SFS greedy forward usando LightGBM como estimador. El SFS evalúa, en cada paso, qué feature agrega más información útil para la predictibilidad del mercado usando validación cruzada temporal (TimeSeriesSplit). De esta forma el modelo selecciona automáticamente las features más relevantes del pool de 45 candidatas.

**Meta-labeling (dos modelos).** Se pasó de un único clasificador a una arquitectura de dos modelos:
- **M1** decide si hay trade o no hay trade (activo vs neutral)
- **M2** decide la dirección del trade (la apuesta del EMA será correcta o no)

Cada modelo recibió su propio conjunto de features, seleccionadas de forma independiente por el SFS con LGBM. Esto permite que cada modelo se especialice en una tarea distinta.

**Side en el Triple Barrier Method.** Al implementar meta-labeling y no ver mejoras en las métricas, se identificó que el problema estaba en el labeling. El Triple Barrier Method original generaba labels simétricos ({+1, -1, 0}) sin tener en cuenta la dirección de la apuesta. Se incorporó el parámetro `side` usando la señal del cruce de EMAs (EMA9/EMA21), de forma que las barreras se orientan en la dirección de cada apuesta. Así, el label +1 significa "la apuesta del EMA fue correcta" y el label -1 significa "la apuesta falló", lo que alinea el labeling con lo que los modelos necesitan aprender.

---

## Resultados (v6 — versión actual)

Split temporal 80/20. Test: abril 2025 → abril 2026. Capital inicial $10,000. Trade size $500. Fee 0.1% por lado.

| Modelo | Net P&L | Sharpe | Max DD | Trades | % Flat |
|--------|---------|--------|--------|--------|--------|
| XGB + vol filter | **+$333** | **1.448** | -2.90% | 64 | 73.6% |
| MLP + vol filter | **+$426** | **1.561** | -2.89% | 77 | 58.4% |
| LGBM + vol filter | **+$222** | **0.980** | -2.90% | 62 | 74.7% |
| XGB (sin filtro) | +$269 | 0.819 | -4.21% | 123 | 25.3% |
| **Buy & Hold** | **-$3,408** | 0.167 | -74.82% | 1 | 0.0% |

El filtro de régimen de volatilidad (FLAT cuando la vol rolling 72h < mediana del train) mejora el Sharpe de ~0.8 a ~1.5 y reduce el MaxDD a la mitad.

---

## Arquitectura v6

### Pipeline de dos modelos (Meta-Labeling — López de Prado cap. 3/4)

```
EMA9/EMA21 crossover  →  señal primaria (LONG o SHORT)
        │
        ▼
Triple Barrier Method  →  labels: ¿fue la apuesta EMA correcta? {+1, 0, -1}
        │
        ├── M1 (LGBM/XGB/MLP)  →  ¿vale la pena apostar? P(activo)
        │
        └── M2 (LGBM/XGB/MLP)  →  ¿la apuesta EMA saldrá bien? P(correcto | activo)
                │
                ▼
        P(+1) = P(activo) × P(correcto)   [si EMA dice LONG]
        P(-1) = P(activo) × P(correcto)   [si EMA dice SHORT]
        P(0)  = 1 - P(activo)
                │
                ▼
        Filtro de vol: FLAT si vol_rolling < mediana_train
```

El modelo no predice si el precio va a subir o bajar directamente. Aprende a identificar cuándo la señal del cruce EMA es confiable.

### Feature Selection: SFS doble con LGBM

Dos SFS independientes (greedy forward, TimeSeriesSplit 5-fold, f1_macro):

**M1** — activo vs neutral (k=11, f1_CV=0.550):
`return_skew_72h, unique_senders, log_return_roll_std72, macd_histogram, fee_apr_proxy, price_divergence, rsi_14, body_ratio, stoch_d, amihud_illiquidity, bvc_imbalance`

**M2** — apuesta correcta vs incorrecta (k=7, f1_CV=0.511):
`log_return_lag4, bvc_imbalance, net_flow_proxy, macd_histogram, donchian_pos, fee_apr_proxy, volumeUSD_roll_std72`

Los features de microestructura (`bvc_imbalance`, `amihud_illiquidity`) fueron seleccionados por el SFS — aparecen en ambos modelos.

### Labeling: Triple Barrier con Side

```
ptSl = [2.0, 1.0]   →  PT = 2× volatilidad, SL = 1× volatilidad (ratio 2:1)
t1   = 12 barras    →  horizonte máximo 48h
side = EMA9/EMA21   →  barreras orientadas a la dirección de la apuesta

label +1  →  PT tocado primero (EMA correcta, ganó)
label -1  →  SL tocado primero (EMA equivocada, perdió)
label 0   →  expiró en t1 sin tocar barreras
```

Distribución resultante: **+1=31.8%, 0=11.3%, -1=56.9%**. El cruce EMA falla el 57% de las veces — el trabajo de M2 es filtrar cuándo confiar en él.

### Filtro de régimen de volatilidad

El análisis por régimen mostró que los modelos son rentables en alta volatilidad y pierden en baja:

| Régimen | LGBM | XGB | BH |
|---------|------|-----|----|
| Alta vol (369 eventos) | +$229 (Sharpe 0.978) | +$185 (Sharpe 0.791) | -$2,552 |
| Baja vol (381 eventos) | -$66 (Sharpe -0.334) | -$12 (Sharpe -0.034) | -$3,408 |

Threshold calculado sobre el train set (sin lookahead): mediana de `log_return_roll_std72`.

---

## Fuentes de Datos

| Fuente | Granularidad | Datos obtenidos |
|--------|-------------|-----------------|
| **Binance API** | 1h | OHLCV histórico de UNI/USDT (~5 años) |
| **The Graph** | 1h | Métricas del protocolo Uniswap v3: volumen DEX, TVL, precio on-chain, fees |
| **Dune Analytics** | 1h | Actividad de transferencias UNI: cantidad, volumen, wallets únicas, actividad de ballenas |
| **Dollar Bars** | Variable | Duración promedio de barras de $6.9M de volumen negociado |

### 45 features candidatas

El pool de features que entra al SFS incluye:
- **Técnicas**: RSI, MACD, Bollinger Bands, ATR, Stochastic, CCI, Williams %R, ROC, OBV, MFI, EMA ratio, ADX, Donchian, VWAP deviation
- **Direccionales (para M2)**: body_ratio, wicks, bb_pct_b, lags cortos (1-8 barras)
- **On-chain**: fee_apr_proxy, tvl_change_pct, unique_senders, net_flow_proxy, transfer_count, whale_volume_ratio, price_divergence
- **Microestructura (LdP cap. 19)**: BVC imbalance, Amihud illiquidity, Roll's spread

---

## Estructura del Proyecto

```
process_data_v5.py      → Carga, merge horario, resample a 4h, 45+ features
triple_barrier_v4.py    → Triple Barrier con side (EMA) y ptSl asimétrico
meta_labeling_v6.py     → SFS doble, MetaPipeline (M1+M2), predict_proba side-aware
simulate_v4.py          → Simulación LONG/SHORT/FLAT + Buy & Hold + métricas financieras
main_v6.py              → Orquestador principal (punto de entrada)
vol_regime_analysis.py  → Análisis de rendimiento por régimen de volatilidad
feature_selection_v6.py → SFS standalone (multiclase, para exploración)
```

## Uso

```bash
pip install -r requirements.txt

# Pipeline completo (SFS + entrenamiento + simulación)
python3 main_v6.py

# Análisis por régimen de volatilidad
python3 vol_regime_analysis.py
```

Los datos procesados se generan en `data/cache/` y `data/processed/` (ignorados por git).

## Stack Tecnológico

| Herramienta | Rol |
|-------------|-----|
| Python 3.10+ | Lenguaje principal |
| XGBoost / LightGBM | Clasificadores de gradient boosting |
| scikit-learn | MLP, TimeSeriesSplit, métricas |
| pandas / NumPy | Manipulación de datos |
| The Graph API / Dune API / Binance API | Fuentes de datos |
| pyarrow | I/O en formato Parquet |
| matplotlib | Visualizaciones |

---

## Evolución del Proyecto

| Versión | Cambio principal | Mejor Sharpe |
|---------|-----------------|--------------|
| v4 | Triple Barrier + CUSUM, 3 modelos base | ~0.3 |
| v5 | Sin leakage, 10 features via MI+dCor+Spearman | 0.33 |
| v6 | Meta-labeling + side EMA + filtro vol | **1.56** |
