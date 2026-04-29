# Modelo de Riesgo Crediticio

> Predicción de incumplimiento de pago con XGBoost, SMOTE y técnicas de ML aplicadas a datos financieros.  

---

## Contexto del negocio

Las instituciones financieras enfrentan el reto de estimar la probabilidad de que un cliente **incumpla un pago en los próximos 24 meses**. Un modelo robusto permite:

- Optimizar el límite de crédito por cliente
- Reducir la **cartera vencida** (Non-Performing Loans)
- Automatizar decisiones de originación de crédito
- Cumplir con marcos regulatorios como Basilea III / IFRS 9

---

## Dataset

| Columna | Descripción |
|--------|-------------|
| `SeriousDlqin2yrs` | **Target** — Mora grave en 2 años (1 = sí) |
| `RevolvingUtilizationOfUnsecuredLines` | Uso de líneas rotativas (tarjetas) |
| `age` | Edad del solicitante |
| `NumberOfTime30-59DaysPastDueNotWorse` | Retrasos leves (30–59 días) |
| `DebtRatio` | Ratio deuda / ingreso |
| `MonthlyIncome` | Ingreso mensual |
| `NumberOfOpenCreditLinesAndLoans` | Líneas de crédito abiertas |
| `NumberOfTimes90DaysLate` | Retrasos graves (90+ días) |
| `NumberRealEstateLoansOrLines` | Créditos hipotecarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Retrasos moderados (60–89 días) |
| `NumberOfDependents` | Dependientes económicos |

**Desbalance de clases:** ~7% morosos / ~93% no morosos

---

## Stack Tecnológico

```
Python 3.12
├── pandas          → Limpieza y EDA
├── scikit-learn    → Preprocesamiento, métricas, CV
├── imbalanced-learn (SMOTE) → Balanceo de clases
├── xgboost         → Modelo principal
├── matplotlib      → Visualizaciones
└── seaborn         → Gráficas de correlación
└── jupyter notebook → Documentación del proceso
```

---

## Arquitectura del Pipeline

```
Raw Data
   │
   ▼
[EDA & Estadísticas descriptivas]
   │
   ▼
[Limpieza]
  ├── Imputación por mediana (MonthlyIncome, NumberOfDependents)
  └── Cap de outliers en p99
   │
   ▼
[Feature Engineering]
  ├── debt_income_ratio
  ├── total_past_due
  ├── utilization_per_line
  └── income_per_dependent
   │
   ▼
[Train / Test split — 80/20 estratificado]
   │
   ▼
[SMOTE — balanceo de clases en train]
   │
   ▼
[XGBoost Classifier]
  ├── n_estimators: 500
  ├── max_depth: 5
  ├── learning_rate: 0.05
  └── regularización: L1 + L2
   │
   ▼
[Evaluación]
  ├── AUC-ROC
  ├── PR-AUC
  ├── Matriz de Confusión
  ├── Curva ROC
  └── Validación Cruzada 5-Fold
```

---

## Resultados

| Métrica | Valor |
|---------|-------|
| **AUC-ROC (test)** | **0.9844** |
| **PR-AUC (test)** | **0.9137** |
| CV AUC-ROC (5-fold) | 0.9885 ± 0.0012 |
| Recall (Morosos) | 0.92 |
| Precision (Morosos) | 0.62 |

> Un **AUC-ROC de 0.98** indica que el modelo discrimina con alta eficacia entre clientes buenos y malos — rendimiento comparable a modelos en producción de instituciones financieras.

### Visualizaciones generadas

| Archivo | Contenido |
|---------|-----------|
| `01_target_distribution.png` | Desbalance de clases |
| `02_feature_distributions.png` | Distribuciones por clase |
| `03_correlation_matrix.png` | Correlaciones entre variables |
| `04_model_evaluation.png` | ROC, PR, Confusión, Importancia |
| `05_score_distribution.png` | Distribución de scores por clase |
| `06_cross_validation.png` | Estabilidad del modelo en 5 folds |

---

## Cómo ejecutar

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/credit-risk-model
cd credit-risk-model

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python credit_risk_model.py
```

### Con datos reales de Kaggle

```bash
# Descargar dataset (requiere cuenta Kaggle)
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/

# Modificar en credit_risk_model.py la línea de carga:
# df = pd.read_csv("data/cs-training.csv", index_col=0)
```

---

## Estructura del Proyecto

```
riesgo_crediticio/
├── notebooks/                  ← Documentación del proyecto.
│   └── data/
├── data/
│   └── credit_data.csv          ← Dataset generado
├── outputs/
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_matrix.png
│   ├── 04_model_evaluation.png
│   ├── 05_score_distribution.png
│   ├── 06_cross_validation.png
│   ├── eda_stats.csv
│   ├── metrics_summary.csv
│   └── test_scores.csv
├── models/
│   └── xgb_credit_risk.json     ← Modelo serializado
├── generate_data.py             ← Generador de datos sintéticos
├── credit_risk_model.py         ← Pipeline principal
├── requirements.txt
└── README.md
```

---

## Decisiones técnicas clave

**¿Por qué SMOTE y no `scale_pos_weight`?**  
SMOTE genera ejemplos sintéticos en el espacio de features, lo que ayuda al modelo a aprender la frontera de decisión con mayor detalle en la clase minoritaria. Se aplica **solo al conjunto de entrenamiento** para evitar data leakage.

**¿Por qué umbral 0.40 en lugar de 0.50?**  
En riesgo crediticio el costo de un **Falso Negativo** (otorgar crédito a un moroso) supera al de un **Falso Positivo** (rechazar a un buen cliente). Bajar el umbral incrementa el Recall en morosos a costa de precisión, lo que es coherente con el objetivo de negocio.

**¿Por qué AUC-ROC como métrica principal?**  
Es invariante al umbral y adecuada para datasets desbalanceados. Se complementa con **PR-AUC** que es más sensible al desempeño en la clase minoritaria (morosos).

---

## Contexto profesional

Este proyecto refleja la metodología utilizada en el análisis de riesgo crediticio del sector bancario mexicano, alineada con:
- Metodologías internas de originación de crédito (BBVA, Santander, Banorte)
- Lineamientos de la **CNBV** para modelos IRB
- Marco de capital de **Basilea III**

---

## Autor

**Martin Alejos Martínez**  
Analista de datos en el sector de Riesgo | Python · SQL · Power BI  
[LinkedIn](https://www.linkedin.com/in/martin-alejos-martinez-06a6aa192) · [GitHub](https://github.com/Martin-Gato)
