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

## Fuentes de Datos

El primer desafío fue encontrar fuentes de datos gratuitas, confiables y con suficiente granularidad horaria. Se utilizaron cuatro fuentes:

| Fuente | Granularidad | Datos obtenidos |
|--------|-------------|-----------------|
| **Binance API** | 1h | OHLCV histórico de UNI/USDT (~5 años) |
| **The Graph** | 1h | Métricas del protocolo Uniswap v3: volumen DEX, TVL, precio on-chain, fees |
| **Dune Analytics** | 1h | Actividad de transferencias UNI: cantidad, volumen, wallets únicas, actividad de ballenas |
| **Dollar Bars** | Variable | Duración promedio de barras de $6.9M de volumen negociado (~10,000 barras) |

### Dollar Bars

En lugar de usar barras de tiempo fijas, se construyeron **dollar bars**: barras que se completan cada vez que se acumulan $6.9M de volumen negociado en Binance. Este umbral fue elegido para producir aproximadamente 10,000 barras a lo largo del dataset (~4.4h promedio por barra). La feature `dollar_bar_duration` captura cuánto tiempo tardó en completarse la última barra, aportando información sobre la intensidad del flujo de capital independiente del tiempo.

### Descartado: Distribución de Token y Sentimiento

Versiones anteriores del proyecto usaban datos de **distribución de holders** (concentración por ballenas, Herfindahl index, crecimiento de wallets) y el **Fear & Greed Index** obtenidos con granularidad diaria. Estas fuentes fueron descartadas en v5 por introducir **data leakage**: al asignar un valor diario a cada barra de 4h dentro del mismo día, el modelo accedía implícitamente a información del futuro durante el entrenamiento, inflando artificialmente las métricas.

---

## Decisiones de Diseño

### Solo frecuencia 4h

Se evaluaron inicialmente frecuencias 4h y daily. La frecuencia daily fue descartada por **escasez de datos**: con ~1,825 barras diarias disponibles, no hay suficientes muestras para entrenar modelos de ML robustos con validación walk-forward. La frecuencia 4h con ~10,000 barras es la única viable.

### Descarte de LSTM y GRU

Versiones anteriores exploraron redes neuronales recurrentes (LSTM, GRU) para capturar dependencias temporales. Fueron descartadas porque la cantidad de datos disponible es insuficiente para entrenar modelos con esa complejidad de parámetros sin caer en sobreajuste. Se optó por modelos más simples: XGBoost, LightGBM y un MLP de dos capas ocultas (16×3 neuronas).

### Reducción de 180 a 10 features

El proyecto comenzó con ~180 features entre técnicas, on-chain y sentimiento. Se realizó un proceso de selección en tres etapas:

1. **Eliminación por leakage**: se removieron todas las features de granularidad diaria (whale_balance_pct, herfindahl_index, holders_growth, fear_greed_value).
2. **Análisis de correlación no lineal**: se calculó Mutual Information (MI), Distance Correlation (dCor) y Spearman rank correlation entre cada feature y el label.
3. **Filtro de redundancia**: se eliminaron features con correlación de Pearson > 0.85 entre sí, priorizando las de mayor información con el target.

Las features supervivientes presentaron valores de MI y dCor consistentemente por encima de cero, indicando que contienen información potencialmente valiosa para la predicción de la señal de precio.

**Features finales (10):**

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `log_return_lag1` | Binance | Log-retorno de la barra anterior |
| `log_return_lag4` | Binance | Log-retorno de 4 barras atrás (16h) |
| `volatility_24h` | Binance | Volatilidad rolling 6 barras (24h) |
| `high_low_range` | Binance | Rango high-low normalizado por close |
| `log_return_roll_std72` | Binance | Volatilidad rolling 18 barras (72h) |
| `price_divergence` | Binance + The Graph | Divergencia precio CEX vs DEX |
| `volumeUSD_roll_std72` | The Graph | Std del volumen DEX en 72h |
| `transfer_count` | Dune | Transferencias UNI en el período de 4h |
| `whale_volume_ratio` | Dune | Proporción del volumen movido por ballenas |
| `dollar_bar_duration` | Binance (dollar bar) | Duración de la última barra de $6.9M |

---

## Labeling: Triple Barrier Method

Los labels se generan con el **Triple Barrier Method** de López de Prado (AFML):

- **+1** — el precio sube 2× la volatilidad rolling antes de 48h
- **-1** — el precio baja 2× la volatilidad rolling antes de 48h
- **0** — el precio no toca ninguna barrera en 48h (neutral)

Para reducir la correlación entre eventos solapados se aplica un **filtro CUSUM simétrico** que selecciona solo los momentos donde la acumulación de retornos supera la volatilidad promedio, reduciendo los eventos de ~10,900 a ~3,750 más independientes.

Distribución de labels resultante: **+1=38%**, **0=25%**, **-1=37%** — balanceada entre las tres clases.

---

## Resultados

Split temporal 80/20. Capital inicial $10,000. Trade size $500. Fee 0.1% por lado.
El Buy & Hold invierte los $10,000 completos desde el inicio del período de test.

| Modelo | Net P&L | Sharpe | Max DD | Accuracy | F1-macro | Trades | % Flat |
|--------|---------|--------|--------|----------|----------|--------|--------|
| XGB-4h | **+$75.82** | 0.330 | -4.60% | 0.431 | 0.411 | 89 | 44.0% |
| LGBM-4h | +$21.60 | 0.112 | -4.84% | 0.424 | 0.395 | 94 | 42.8% |
| MLP-4h | -$137.84 | -0.538 | -4.97% | 0.421 | 0.409 | 110 | 44.0% |
| **Buy & Hold** | **-$3,408** | 0.167 | -74.82% | — | — | 1 | 0.0% |

Los tres modelos superan al Buy & Hold ampliamente en términos de drawdown máximo (-5% vs -75%). El período de test coincidió con un mercado bajista severo para UNI, lo que hace que el benchmark sea especialmente desfavorable.

Las ganancias absolutas son modestas, lo cual es coherente con labels sin leakage y un problema genuinamente difícil. Con más datos disponibles para backtesting hubiese sido posible validar la robustez de la estrategia en distintos regímenes de mercado, pero el dataset disponible no lo permitía sin comprometer el tamaño del set de entrenamiento.

---

## Arquitectura del Pipeline

```
process_data_v5.py       → Carga, merge horario y resample a 4h
feature_selection_v5.py  → Selección de features (MI + dCor + Spearman + correlación)
triple_barrier_v4.py     → Labeling con Triple Barrier Method + filtro CUSUM
models_v4.py             → XGBoost, LightGBM, MLP con walk-forward
simulate_v4.py           → Simulación LONG/SHORT/FLAT + Buy & Hold
main_v5.py               → Orquestador (punto de entrada)
```

## Uso

```bash
pip install -r requirements.txt

# Correr el pipeline completo
python main_v5.py
```

Los archivos de caché y datos procesados se generan localmente en `data/cache/` y `data/processed/` (ignorados por git).

## Stack Tecnológico

| Herramienta | Rol |
|-------------|-----|
| Python 3.10+ | Lenguaje principal |
| XGBoost / LightGBM | Clasificadores de gradient boosting |
| scikit-learn | MLP, validación walk-forward, métricas |
| pandas / NumPy | Manipulación de datos |
| The Graph API / Dune API / Binance API | Fuentes de datos |
| pyarrow | I/O en formato Parquet |
| matplotlib | Visualizaciones |
