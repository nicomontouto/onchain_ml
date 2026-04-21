# Onchain ML — Pipeline de Trading Algorítmico con Datos On-Chain (UNI/USDT)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189fdd)

## Descripción

Pipeline end-to-end de machine learning para construir y backtestear estrategias de trading algorítmico sobre el par UNI/USDT. El proyecto combina métricas **on-chain del protocolo Uniswap** con **datos de precio** e **indicadores de sentimiento de mercado** para entrenar clasificadores que predicen la dirección del precio en el siguiente período.

El pipeline genera señales LONG/SHORT/FLAT y simula una cuenta de capital real, comparando el desempeño contra un benchmark Buy & Hold.

## Desafío: Obtención de Datos

El primer obstáculo del proyecto fue encontrar fuentes de datos gratuitas, confiables y con suficiente granularidad. Se utilizaron tres fuentes principales:

- **Binance API** — Precio histórico del token UNI con velas de 1 hora (OHLCV), cubriendo aproximadamente 5 años de historia.
- **The Graph** — Métricas del protocolo Uniswap v3 on-chain: TVL, volumen de swaps, ingresos por comisiones, posiciones activas y profundidad de liquidez.
- **Dune Analytics** — Distribución de holders del token UNI: concentración por ballenas, número de wallets activas y evolución de la distribución en el tiempo.
- **Fear & Greed Index** — Indicador de sentimiento del mercado cripto (frecuencia diaria, interpolado a frecuencias intradiarias).

La hipótesis central es que la **distribución del token** —quién tiene UNI y cómo cambia esa concentración— puede ser un predictor relevante del precio en tokens de capitalización media, donde el comportamiento de los grandes tenedores tiene mayor impacto relativo que en tokens de muy alta liquidez.

## Modelos

Se evaluaron dos enfoques de clasificación:

### Gradient Boosting (XGBoost)
Clasificador basado en árboles con optimización por gradiente. Se entrenó con validación walk-forward para evitar data leakage y se realizó búsqueda de hiperparámetros. **Produjo los resultados más consistentes y determinantes del proyecto.**

### Red Neuronal Recurrente (LSTM)
Red de memoria a largo-corto plazo, adecuada para capturar dependencias temporales en series de tiempo financieras. Sin embargo, la cantidad de datos disponibles resultó insuficiente para entrenar un modelo con la complejidad de parámetros empleada, lo que limitó su capacidad de generalización.

## Resultados y Limitaciones

El modelo XGBoost superó al LSTM en todas las frecuencias evaluadas. Se identificaron dos problemas principales que se están trabajando actualmente:

1. **Volumen de datos insuficiente para la LSTM**: La cantidad de muestras disponibles —limitada por el historial del protocolo Uniswap v3 y la frecuencia de las métricas on-chain— no es suficiente para entrenar redes recurrentes con el número de parámetros utilizado sin caer en sobreajuste.

2. **Exceso de features**: El conjunto de features construido es amplio (técnicos + on-chain + sentimiento), lo cual incrementa el riesgo de overfitting, especialmente en la LSTM. Se está trabajando en selección de features y reducción de dimensionalidad para evaluar correctamente el peso real de la distribución del token como predictor en este tipo de activo.

Ver `reporte_uni_ml_v3.pdf` para el informe completo con curvas de equity, drawdowns, métricas de validación y ranking de importancia de features.

## Variantes de Estrategia (V3)

| Variante | Modelo | Lógica de señal |
|----------|--------|----------------|
| A | XGBoost | Umbral de probabilidad fijo |
| B | XGBoost | Umbral dinámico ajustado por Fear & Greed Index |
| C | LSTM | Umbral fijo asimétrico (sesgo largo) |
| BH | — | Buy & Hold (benchmark) |

Frecuencias evaluadas: **4h** y **daily** (1h descartada tras ablación).

## Features Utilizadas

**Precio (técnicos):** retornos, log-retornos, RSI, MACD, Bandas de Bollinger, ATR, ratios de volumen.

**On-chain (Uniswap v3):** TVL, volumen de swaps, ingresos por fees, posiciones activas, profundidad de liquidez.

**Distribución del token (Dune):** concentración de holders, número de wallets activas, participación de grandes tenedores.

**Sentimiento:** Fear & Greed Index (diario, forward-filled a frecuencias intradiarias).

## Arquitectura del Pipeline

```
fetch_data.py          → Descarga de datos (Binance, The Graph, Dune, Fear & Greed)
feature_engineering.py → Construcción de features técnicas y on-chain
process_data.py        → Limpieza, merge y etiquetado (target: signo del retorno)
model_baseline.py      → XGBoost con validación walk-forward y tuning
model_lstm.py          → Entrenamiento de la LSTM
simulate_v3.py         → Simulación LONG/SHORT/FLAT con capital real en USD
generate_report_v3.py  → Generación de reporte PDF
main_v3.py             → Orquestador principal (corre el pipeline completo)
```

## Estructura del Proyecto

```
onchain_ml/
├── fetch_data.py           # Ingesta de datos
├── feature_engineering.py  # Construcción de features
├── process_data.py         # Limpieza y etiquetado
├── model_baseline.py       # XGBoost walk-forward + tuning
├── model_lstm.py           # Entrenamiento LSTM
├── evaluate.py             # Métricas de evaluación
├── simulate_v3.py          # Backtester V3 LONG/SHORT/FLAT
├── generate_report_v3.py   # Generación de reporte PDF
├── main.py                 # Orquestador V2
├── main_v3.py              # Orquestador V3 (punto de entrada recomendado)
├── best_lstm.pt            # Pesos del modelo LSTM entrenado
├── reporte_uni_ml_v3.pdf   # Reporte de resultados completo (V3)
├── data/
│   ├── thegraph_uni_protocol.csv   # Snapshot de métricas on-chain
│   ├── dune_uni_holders.csv        # Snapshot de distribución de holders
│   ├── figures/                    # Gráficos de resultados V2
│   └── figures_v3/                 # Gráficos de resultados V3
└── requirements.txt
```

## Uso

```bash
pip install -r requirements.txt

# Correr el pipeline completo
python main_v3.py

# Saltear el reentrenamiento de XGBoost (usa probabilidades cacheadas)
python main_v3.py --skip-xgb-retrain

# Forzar reentrenamiento completo
python main_v3.py --no-cache
```

Los archivos de caché y datos procesados se generan localmente en `data/cache/` y `data/processed/` (ignorados por git).

## Stack Tecnológico

| Herramienta | Rol |
|-------------|-----|
| Python 3.10+ | Lenguaje principal |
| XGBoost | Clasificador de gradient boosting |
| PyTorch | Modelo LSTM recurrente |
| scikit-learn | Validación walk-forward y métricas |
| pandas / NumPy | Manipulación de datos |
| requests / pyarrow | Llamadas a APIs y I/O en formato Parquet |
| fpdf2 | Generación de reportes PDF |
| matplotlib | Visualizaciones |

## Trabajo Futuro

- Selección y reducción de features para aislar el poder predictivo de la distribución del token.
- Aumentar el volumen de datos históricos o explorar técnicas de data augmentation para mejorar el entrenamiento de la LSTM.
- Evaluar arquitecturas más simples (GRU, modelos lineales con memoria) como baseline para la componente temporal.
- Extender el análisis a otros tokens de capitalización similar con mayor historia on-chain.
- Integrar feed en tiempo real de Binance para paper trading.
