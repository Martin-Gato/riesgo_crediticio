"""
Genera datos sintéticos sobre riesgo créditicio. 
Autor : Martin Alejos Martínez
"""

import numpy as np
import pandas as pd

def generate_credit_data(n_samples: int = 150_000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Clientes buenos (~93%) y malos (~7%) — distribución real del dataset
    n_bad  = int(n_samples * 0.0700)
    n_good = n_samples - n_bad

    def _make_group(n, is_bad):
        """Genera features con distribuciones distintas por segmento."""
        inc_mu = 3_800 if is_bad else 6_200          # ingreso mensual USD
        age_mu = 40    if is_bad else 47

        utilization = rng.beta(2 if is_bad else 0.8, 2, n).clip(0, 1)
        age         = rng.normal(age_mu, 9, n).clip(18, 90).astype(int)
        income      = rng.lognormal(np.log(inc_mu), 0.6, n)
        income      = np.where(rng.random(n) < 0.02, np.nan, income)   # 2% missing

        debt_ratio  = rng.beta(1.5 if is_bad else 0.5, 3, n).clip(0, 1)

        past30_59   = rng.poisson(1.5 if is_bad else 0.1, n)
        past60_89   = rng.poisson(0.8 if is_bad else 0.05, n)
        past90plus  = rng.poisson(0.5 if is_bad else 0.02, n)

        open_lines  = rng.poisson(8, n).clip(0, 40)
        real_estate = rng.poisson(1 if is_bad else 1.5, n).clip(0, 10)
        dependents  = rng.poisson(1.0, n).clip(0, 10)
        dependents  = np.where(rng.random(n) < 0.025, np.nan, dependents)  # 2.5% missing

        return pd.DataFrame({
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": past90plus,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": past60_89,
            "NumberOfDependents": dependents,
            "SeriousDlqin2yrs": int(is_bad),
        })

    df = pd.concat([_make_group(n_good, False), _make_group(n_bad, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_credit_data()
    df.to_csv("data/credit_data.csv", index=False)
    print(f"Dataset generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Tasa de morosidad: {df['SeriousDlqin2yrs'].mean():.2%}")
