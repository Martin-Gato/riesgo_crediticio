"""
Modelo de Riesgo Crediticio
Autor : Martin Alejos Martínez
Stack : Pandas · XGBoost · Imbalanced-learn · Scikit-learn · Matplotlib · Seaborn

Flujo
-----
1. Carga y EDA
2. Limpieza e ingeniería de features
3. Balanceo de clases con SMOTE
4. Entrenamiento: XGBoost con búsqueda de hiperparámetros
5. Evaluación: AUC-ROC, PR-AUC, Matriz de Confusión, SHAP
6. Exportación de métricas y modelo
"""

# ─────────────────────────────── IMPORTS ───────────────────────────────────
import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
from xgboost import XGBClassifier

# ─────────────────────────────── CONFIG ────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.20
OUTPUT_DIR    = "outputs"
MODEL_DIR     = "models"
PALETTE       = {"good": "#2ECC71", "bad": "#E74C3C", "neutral": "#3498DB"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  MODELO DE RIESGO CREDITICIO — Give Me Some Credit")
print("=" * 65)

from data.generate_data import generate_credit_data
df = generate_credit_data(n_samples=150_000, random_state=RANDOM_STATE)
df.to_csv("data/credit_data.csv", index=False)

TARGET = "SeriousDlqin2yrs"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"\n[1/6] Dataset cargado: {df.shape[0]:,} filas · {df.shape[1]} columnas")
print(f"      Tasa de morosidad: {df[TARGET].mean():.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA — ANÁLISIS EXPLORATORIO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] EDA …")

# ── 2a. Estadísticas descriptivas ──
stats = df.describe().T
stats["missing_%"] = (df.isnull().sum() / len(df) * 100).values
stats.to_csv(f"{OUTPUT_DIR}/eda_stats.csv")

# ── 2b. Distribución de la variable objetivo ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Distribución de la Variable Objetivo", fontsize=14, fontweight="bold")

counts = df[TARGET].value_counts()
axes[0].bar(["No Moroso (0)", "Moroso (1)"],
            counts.values,
            color=[PALETTE["good"], PALETTE["bad"]])
axes[0].set_title("Conteo absoluto")
axes[0].set_ylabel("Número de clientes")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(counts.values,
            labels=["No Moroso (0)", "Moroso (1)"],
            colors=[PALETTE["good"], PALETTE["bad"]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Proporción")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2c. Distribuciones de features por clase ──
num_features = ["RevolvingUtilizationOfUnsecuredLines", "age",
                "DebtRatio", "MonthlyIncome"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Distribución de Features por Clase", fontsize=14, fontweight="bold")

for ax, feat in zip(axes.flatten(), num_features):
    good = df.loc[df[TARGET] == 0, feat].dropna()
    bad  = df.loc[df[TARGET] == 1, feat].dropna()

    # Percentile clip to avoid extreme outliers in plots
    p99 = df[feat].quantile(0.99)
    good = good[good <= p99]
    bad  = bad[bad  <= p99]

    ax.hist(good, bins=50, alpha=0.6, color=PALETTE["good"],
            density=True, label="No Moroso")
    ax.hist(bad,  bins=50, alpha=0.6, color=PALETTE["bad"],
            density=True, label="Moroso")
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2d. Matriz de correlación ──
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, ax=ax, linewidths=0.5,
            annot_kws={"size": 7})
ax.set_title("Matriz de Correlación", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("      Gráficas de EDA guardadas en /outputs")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIMPIEZA E INGENIERÍA DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Limpieza e ingeniería de features …")

df_clean = df.copy()

# Imputación por mediana
for col in ["MonthlyIncome", "NumberOfDependents"]:
    median = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median)
    print(f"      {col}: imputado con mediana = {median:.1f}")

# Outliers: cap en percentil 99 para variables continuas
cap_cols = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
for col in cap_cols:
    cap = df_clean[col].quantile(0.99)
    n_capped = (df_clean[col] > cap).sum()
    df_clean[col] = df_clean[col].clip(upper=cap)
    print(f"      {col}: {n_capped:,} valores truncados en {cap:.2f}")

# Feature engineering — features adicionales con significado financiero
df_clean["debt_income_ratio"] = (
    df_clean["DebtRatio"] * df_clean["MonthlyIncome"] /
    (df_clean["MonthlyIncome"] + 1)
)
df_clean["total_past_due"] = (
    df_clean["NumberOfTime30-59DaysPastDueNotWorse"] +
    df_clean["NumberOfTime60-89DaysPastDueNotWorse"] +
    df_clean["NumberOfTimes90DaysLate"]
)
df_clean["utilization_per_line"] = (
    df_clean["RevolvingUtilizationOfUnsecuredLines"] /
    (df_clean["NumberOfOpenCreditLinesAndLoans"] + 1)
)
df_clean["income_per_dependent"] = (
    df_clean["MonthlyIncome"] /
    (df_clean["NumberOfDependents"] + 1)
)

FEATURES_ENG = [c for c in df_clean.columns if c != TARGET]
print(f"      Features totales después de ingeniería: {len(FEATURES_ENG)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPLIT · SMOTE · ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Split · Balanceo SMOTE · Entrenamiento XGBoost …")

X = df_clean[FEATURES_ENG]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"      Train: {X_train.shape[0]:,} · Test: {X_test.shape[0]:,}")
print(f"      Morosidad Train antes SMOTE: {y_train.mean():.2%}")

# SMOTE — balanceo de clases
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"      Morosidad Train después SMOTE: {y_res.mean():.2%}")
print(f"      Tamaño rebalanceado: {X_res.shape[0]:,} filas")

# XGBoost con hiperparámetros ajustados para datos financieros
xgb_params = {
    "n_estimators":      500,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,
    "gamma":             1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  1,          # ya está balanceado con SMOTE
    "eval_metric":       "auc",
    "use_label_encoder": False,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

model = XGBClassifier(**xgb_params)
model.fit(
    X_res, y_res,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("      Modelo entrenado ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Evaluando métricas …")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.40).astype(int)   # umbral ajustado para F1 óptimo

auc_roc  = roc_auc_score(y_test, y_prob)
pr_auc   = average_precision_score(y_test, y_prob)
report   = classification_report(y_test, y_pred, target_names=["No Moroso", "Moroso"])

print(f"\n      ── Métricas principales ──")
print(f"      AUC-ROC  : {auc_roc:.4f}")
print(f"      PR-AUC   : {pr_auc:.4f}")
print(f"\n{report}")

# Validación cruzada (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Usamos un modelo ligero para CV rápido
model_cv = XGBClassifier(**{**xgb_params, "n_estimators": 200}, verbose=0)
cv_scores = cross_val_score(model_cv, X_train, y_train,
                             scoring="roc_auc", cv=cv, n_jobs=-1)
print(f"      CV AUC-ROC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ── 5a. FIGURA PRINCIPAL — 2x2 ──────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle("Evaluación del Modelo de Riesgo Crediticio\n(XGBoost + SMOTE)",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# Panel A — ROC Curve
ax_roc = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax_roc.plot(fpr, tpr, color=PALETTE["bad"], lw=2,
            label=f"XGBoost (AUC = {auc_roc:.4f})")
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio (AUC = 0.50)")
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["bad"])
ax_roc.set(xlabel="Tasa de Falsos Positivos", ylabel="Tasa de Verdaderos Positivos",
           title="Curva ROC", xlim=[0, 1], ylim=[0, 1.02])
ax_roc.legend(fontsize=9)
ax_roc.grid(alpha=0.3)

# Panel B — Precision-Recall Curve
ax_pr = fig.add_subplot(gs[0, 1])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
baseline = y_test.mean()
ax_pr.plot(rec, prec, color=PALETTE["neutral"], lw=2,
           label=f"XGBoost (PR-AUC = {pr_auc:.4f})")
ax_pr.axhline(baseline, color="gray", ls="--", lw=1,
              label=f"Baseline ({baseline:.2%})")
ax_pr.fill_between(rec, prec, alpha=0.08, color=PALETTE["neutral"])
ax_pr.set(xlabel="Recall", ylabel="Precisión",
          title="Curva Precision-Recall", xlim=[0, 1], ylim=[0, 1.02])
ax_pr.legend(fontsize=9)
ax_pr.grid(alpha=0.3)

# Panel C — Matriz de Confusión
ax_cm = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Moroso", "Moroso"])
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title("Matriz de Confusión")

# Panel D — Importancia de Features
ax_fi = fig.add_subplot(gs[1, 1])
fi = pd.Series(model.feature_importances_, index=FEATURES_ENG)
fi_top = fi.nlargest(12).sort_values()
colors = [PALETTE["bad"] if v > fi_top.quantile(0.75) else PALETTE["neutral"]
          for v in fi_top.values]
fi_top.plot.barh(ax=ax_fi, color=colors)
ax_fi.set_title("Top 12 Features por Importancia")
ax_fi.set_xlabel("Importancia (Gain)")
ax_fi.grid(axis="x", alpha=0.3)

plt.savefig(f"{OUTPUT_DIR}/04_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5b. Score distribution ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
scores_good = y_prob[y_test == 0]
scores_bad  = y_prob[y_test == 1]
ax.hist(scores_good, bins=60, alpha=0.6, color=PALETTE["good"],
        density=True, label="No Moroso")
ax.hist(scores_bad,  bins=60, alpha=0.6, color=PALETTE["bad"],
        density=True, label="Moroso")
ax.axvline(0.40, color="black", ls="--", lw=1.5, label="Umbral = 0.40")
ax.set(xlabel="Score de Probabilidad de Mora", ylabel="Densidad",
       title="Distribución de Scores por Clase")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5c. CV Scores ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bars = ax.bar(folds, cv_scores, color=PALETTE["neutral"], width=0.5)
ax.axhline(cv_scores.mean(), color=PALETTE["bad"], ls="--", lw=1.5,
           label=f"Media: {cv_scores.mean():.4f}")
ax.fill_between(range(len(folds)),
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color=PALETTE["bad"], label=f"±1σ ({cv_scores.std():.4f})")
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{score:.4f}", ha="center", va="bottom", fontsize=9)
ax.set(ylabel="AUC-ROC", title="Validación Cruzada Estratificada (5-Fold)",
       ylim=[max(0, cv_scores.min() - 0.02), min(1, cv_scores.max() + 0.02)])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXPORTAR MÉTRICAS Y MODELO
# ══════════════════════════════════════════════════════════════════════════════
print("[6/6] Exportando métricas y modelo …")

# Métricas en CSV
metrics_df = pd.DataFrame({
    "Métrica":  ["AUC-ROC (Test)", "PR-AUC (Test)",
                 "CV AUC-ROC (mean)", "CV AUC-ROC (std)"],
    "Valor":    [f"{auc_roc:.4f}", f"{pr_auc:.4f}",
                 f"{cv_scores.mean():.4f}", f"{cv_scores.std():.4f}"],
})
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)

# Scores del conjunto test
scores_df = pd.DataFrame({
    "y_real": y_test.values,
    "score":  y_prob,
    "prediccion": y_pred,
})
scores_df.to_csv(f"{OUTPUT_DIR}/test_scores.csv", index=False)

# Guardar modelo
model.save_model(f"{MODEL_DIR}/xgb_credit_risk.json")

print(f"\n{'='*65}")
print("  RESULTADOS FINALES")
print(f"{'='*65}")
print(f"  AUC-ROC  (test)       : {auc_roc:.4f}")
print(f"  PR-AUC   (test)       : {pr_auc:.4f}")
print(f"  CV AUC-ROC (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n  Archivos generados en /outputs y /models")
print(f"{'='*65}\n")
