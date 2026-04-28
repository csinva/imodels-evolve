import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('teachingratings.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Correlation between beauty and eval
corr, pval = stats.pearsonr(df['beauty'], df['eval'])
print(f"\nPearson correlation beauty-eval: r={corr:.4f}, p={pval:.6f}")

# Simple OLS regression
X = sm.add_constant(df['beauty'])
model = sm.OLS(df['eval'], X).fit()
print("\nSimple OLS: beauty -> eval")
print(model.summary())

# Multiple regression controlling for other factors
df_encoded = pd.get_dummies(df, columns=['minority','gender','credits','division','native','tenure'], drop_first=True).astype(float)
controls = ['beauty', 'age', 'students', 'allstudents',
            'minority_yes', 'gender_male', 'credits_single',
            'division_upper', 'native_yes', 'tenure_yes']
controls = [c for c in controls if c in df_encoded.columns]

X_multi = sm.add_constant(df_encoded[controls].astype(float))
model_multi = sm.OLS(df['eval'], X_multi).fit()
print("\nMultiple OLS: beauty + controls -> eval")
print(model_multi.summary())

beauty_coef = model_multi.params['beauty']
beauty_pval = model_multi.pvalues['beauty']
print(f"\nBeauty coefficient: {beauty_coef:.4f}, p-value: {beauty_pval:.6f}")

# Split into high/low beauty and compare evals
median_beauty = df['beauty'].median()
high_beauty = df[df['beauty'] > median_beauty]['eval']
low_beauty = df[df['beauty'] <= median_beauty]['eval']
t_stat, t_pval = stats.ttest_ind(high_beauty, low_beauty)
print(f"\nT-test high vs low beauty groups: t={t_stat:.4f}, p={t_pval:.6f}")
print(f"Mean eval high beauty: {high_beauty.mean():.4f}, low beauty: {low_beauty.mean():.4f}")

# Determine response score
# Strong positive correlation, statistically significant -> high score
if beauty_pval < 0.05 and beauty_coef > 0:
    response = 82
    explanation = (
        f"Beauty has a statistically significant positive impact on teaching evaluations. "
        f"Simple correlation: r={corr:.3f} (p={pval:.4f}). "
        f"In a multiple regression controlling for age, gender, minority status, tenure, and other factors, "
        f"the beauty coefficient is {beauty_coef:.4f} (p={beauty_pval:.4f}), indicating that a one-unit "
        f"increase in beauty score is associated with a {beauty_coef:.4f} point increase in teaching evaluation. "
        f"High-beauty instructors received mean eval {high_beauty.mean():.3f} vs {low_beauty.mean():.3f} for low-beauty (t-test p={t_pval:.4f}). "
        f"The effect is robust to controls, confirming beauty positively influences teaching evaluations."
    )
elif beauty_pval < 0.05 and beauty_coef < 0:
    response = 20
    explanation = (
        f"Beauty has a statistically significant negative impact on teaching evaluations (coef={beauty_coef:.4f}, p={beauty_pval:.4f}). "
        f"This is an unexpected direction."
    )
else:
    response = 25
    explanation = (
        f"Beauty does not have a statistically significant impact on teaching evaluations after controlling for other factors "
        f"(coef={beauty_coef:.4f}, p={beauty_pval:.4f})."
    )

print(f"\nFinal response: {response}")
print(f"Explanation: {explanation}")

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclustion.txt written successfully.")
