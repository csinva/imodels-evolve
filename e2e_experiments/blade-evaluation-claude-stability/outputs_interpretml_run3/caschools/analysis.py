import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('caschools.csv')

# Calculate student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Academic performance: use average of read and math
df['score'] = (df['read'] + df['math']) / 2

print("Dataset shape:", df.shape)
print("\nStudent-teacher ratio stats:")
print(df['str'].describe())
print("\nScore stats:")
print(df['score'].describe())

# Correlation
corr_read, p_read = stats.pearsonr(df['str'], df['read'])
corr_math, p_math = stats.pearsonr(df['str'], df['math'])
corr_score, p_score = stats.pearsonr(df['str'], df['score'])

print(f"\nCorrelation STR vs read: {corr_read:.4f}, p={p_read:.4e}")
print(f"Correlation STR vs math: {corr_math:.4f}, p={p_math:.4e}")
print(f"Correlation STR vs avg score: {corr_score:.4f}, p={p_score:.4e}")

# OLS regression: simple
X = sm.add_constant(df['str'])
model_read = sm.OLS(df['read'], X).fit()
model_math = sm.OLS(df['math'], X).fit()
model_score = sm.OLS(df['score'], X).fit()

print("\n--- OLS: STR -> avg score ---")
print(f"Coef STR: {model_score.params['str']:.4f}, p={model_score.pvalues['str']:.4e}")
print(f"R2: {model_score.rsquared:.4f}")

# Multiple regression controlling for confounders
features = ['str', 'calworks', 'lunch', 'income', 'english', 'expenditure']
df_clean = df[features + ['score']].dropna()
X_multi = sm.add_constant(df_clean[features])
model_multi = sm.OLS(df_clean['score'], X_multi).fit()
print("\n--- Multiple OLS: STR -> score (controlling confounders) ---")
print(f"Coef STR: {model_multi.params['str']:.4f}, p={model_multi.pvalues['str']:.4e}")
print(f"R2: {model_multi.rsquared:.4f}")

# Split into high/low STR groups and t-test
median_str = df['str'].median()
high_str = df[df['str'] > median_str]['score']
low_str = df[df['str'] <= median_str]['score']
t_stat, p_ttest = stats.ttest_ind(low_str, high_str)
print(f"\nT-test (low vs high STR): t={t_stat:.4f}, p={p_ttest:.4e}")
print(f"Mean score low STR: {low_str.mean():.2f}, high STR: {high_str.mean():.2f}")

# Decision tree for feature importance
from sklearn.preprocessing import LabelEncoder
df2 = df.copy()
df2 = df2[['str','calworks','lunch','income','english','expenditure','score']].dropna()
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(df2[['str','calworks','lunch','income','english','expenditure']], df2['score'])
feat_names = ['str','calworks','lunch','income','english','expenditure']
importances = dict(zip(feat_names, dt.feature_importances_))
print("\nDecision Tree feature importances:")
for k,v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

# Conclusion
# Simple regression shows negative correlation (more STR -> lower scores)
# Check significance and direction
str_coef = model_multi.params['str']
str_p = model_multi.pvalues['str']
simple_corr = corr_score
simple_p = p_score

print(f"\nSimple correlation: {simple_corr:.4f} (p={simple_p:.4e})")
print(f"Multiple regression STR coef: {str_coef:.4f} (p={str_p:.4e})")

# If coef is negative and p < 0.05 in simple regression -> yes association
# Multiple regression may lose significance due to confounders
if simple_p < 0.05 and simple_corr < 0:
    simple_sig = True
else:
    simple_sig = False

if str_p < 0.05 and str_coef < 0:
    multi_sig = True
else:
    multi_sig = False

print(f"\nSimple significant negative: {simple_sig}")
print(f"Multiple significant negative: {multi_sig}")

# Score: if simple regression is significant, moderate-to-strong yes
# If only simple (not multiple), somewhat yes
# If neither, no
if simple_sig and multi_sig:
    response = 80
    explanation = (
        f"Yes. Both simple and multiple regression show a statistically significant negative relationship "
        f"between student-teacher ratio (STR) and academic performance (avg test score). "
        f"Simple correlation: r={simple_corr:.3f} (p={simple_p:.2e}). "
        f"In multiple regression controlling for income, lunch subsidies, English learners, calworks, expenditure, "
        f"STR coefficient = {str_coef:.3f} (p={str_p:.2e}). "
        f"Districts with lower STR score higher on average (low-STR mean={low_str.mean():.1f}, high-STR mean={high_str.mean():.1f}). "
        f"The evidence strongly supports that a lower student-teacher ratio is associated with higher academic performance."
    )
elif simple_sig and not multi_sig:
    response = 55
    explanation = (
        f"Weak yes. Simple correlation shows a statistically significant negative relationship "
        f"between STR and academic performance (r={simple_corr:.3f}, p={simple_p:.2e}), "
        f"but this becomes non-significant in multiple regression controlling for confounders "
        f"(STR coef={str_coef:.3f}, p={str_p:.2e}). "
        f"The raw association exists, but socioeconomic confounders (income, lunch subsidies) explain much of it. "
        f"There is a modest positive signal but the causal link is unclear."
    )
else:
    response = 20
    explanation = (
        f"No. The relationship between student-teacher ratio and academic performance is not statistically significant. "
        f"Simple correlation: r={simple_corr:.3f} (p={simple_p:.2e}). "
        f"STR does not appear to be independently associated with test scores."
    )

conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
print(json.dumps(conclusion, indent=2))
