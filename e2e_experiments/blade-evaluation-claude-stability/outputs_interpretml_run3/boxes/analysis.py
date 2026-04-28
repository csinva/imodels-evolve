import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nValue counts for y:")
print(df['y'].value_counts())

# Create binary outcome: chose majority (y==2)
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\nOverall majority choice rate:", df['chose_majority'].mean())
print("\nMajority choice rate by age:")
age_majority = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print(age_majority)

print("\nMajority choice rate by culture:")
culture_majority = df.groupby('culture')['chose_majority'].agg(['mean', 'count'])
print(culture_majority)

# --- Statistical tests ---

# 1. Correlation between age and majority choice
r, p_corr = stats.pointbiserialr(df['age'], df['chose_majority'])
print(f"\nPoint-biserial correlation age vs majority: r={r:.4f}, p={p_corr:.4f}")

# 2. Logistic regression: majority choice ~ age + culture + age*culture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# One-hot encode culture
df_model = pd.get_dummies(df[['chose_majority', 'age', 'culture', 'majority_first']], columns=['culture'])
X = df_model.drop('chose_majority', axis=1).astype(float)
y = df_model['chose_majority']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)
print("\nLogistic regression coefficients:")
for feat, coef in zip(X.columns, lr.coef_[0]):
    print(f"  {feat}: {coef:.4f}")

# 3. OLS regression with statsmodels for p-values
X_sm = sm.add_constant(df[['age', 'majority_first']].astype(float))
ols = sm.OLS(df['chose_majority'], X_sm).fit()
print("\nOLS regression summary:")
print(ols.summary())

# 4. Age trend test: split into younger (4-9) vs older (10-14)
young = df[df['age'] <= 9]['chose_majority']
old = df[df['age'] >= 10]['chose_majority']
t_stat, p_age_group = stats.ttest_ind(young, old)
print(f"\nYoung (4-9) majority rate: {young.mean():.4f} (n={len(young)})")
print(f"Old (10-14) majority rate: {old.mean():.4f} (n={len(old)})")
print(f"t-test: t={t_stat:.4f}, p={p_age_group:.4f}")

# 5. ANOVA across cultures for majority choice
culture_groups = [df[df['culture'] == c]['chose_majority'].values for c in df['culture'].unique()]
f_stat, p_anova = stats.f_oneway(*culture_groups)
print(f"\nANOVA across cultures: F={f_stat:.4f}, p={p_anova:.4f}")

# 6. Interaction: age trend within each culture
print("\nAge-majority correlation within each culture:")
age_culture_corrs = {}
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    if len(sub) > 10:
        r_c, p_c = stats.pointbiserialr(sub['age'], sub['chose_majority'])
        age_culture_corrs[c] = (r_c, p_c, len(sub))
        print(f"  Culture {c}: r={r_c:.4f}, p={p_c:.4f}, n={len(sub)}")

# 7. EBM model from interpret
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=42)
    X_ebm = df[['age', 'culture', 'majority_first', 'gender']].astype(float)
    ebm.fit(X_ebm, df['chose_majority'])
    print("\nEBM feature importances:")
    for name, imp in zip(X_ebm.columns, ebm.term_importances()):
        print(f"  {name}: {imp:.4f}")
except Exception as e:
    print(f"EBM failed: {e}")

# --- Conclusion ---
# Key findings:
# 1. Overall majority choice rate > 50%?
overall_rate = df['chose_majority'].mean()
# 2. Significant age effect?
age_sig = p_corr < 0.05
# 3. Significant culture effect?
culture_sig = p_anova < 0.05
# 4. Direction of age effect
age_direction = "increases" if r > 0 else "decreases"

print(f"\n=== SUMMARY ===")
print(f"Overall majority choice rate: {overall_rate:.3f}")
print(f"Age correlation: r={r:.3f}, p={p_corr:.4f}, significant={age_sig}")
print(f"Culture ANOVA: F={f_stat:.3f}, p={p_anova:.4f}, significant={culture_sig}")
print(f"Age group test: young={young.mean():.3f}, old={old.mean():.3f}, p={p_age_group:.4f}")

# Score: question asks if majority preference DEVELOPS (increases) with age across cultures
# Strong yes if age effect is significant AND positive AND culture modulates it
if age_sig and r > 0 and culture_sig:
    response = 75
    explanation = (
        f"Children's reliance on majority preference significantly increases with age "
        f"(point-biserial r={r:.3f}, p={p_corr:.4f}) and varies across cultural contexts "
        f"(ANOVA F={f_stat:.3f}, p={p_anova:.4f}). Older children (10-14) choose the majority "
        f"at rate {old.mean():.2f} vs {young.mean():.2f} for younger children (4-9). "
        f"Both age and culture have statistically significant effects on majority preference, "
        f"supporting the view that majority-following develops with age across different cultural contexts."
    )
elif age_sig and r > 0:
    response = 65
    explanation = (
        f"Age significantly predicts majority choice (r={r:.3f}, p={p_corr:.4f}), "
        f"but cultural differences are not statistically significant (p={p_anova:.4f}). "
        f"Overall there is a positive developmental trend in majority preference."
    )
elif age_sig and r < 0:
    response = 25
    explanation = (
        f"Age is significantly correlated with majority choice (r={r:.3f}, p={p_corr:.4f}), "
        f"but in the negative direction — older children rely LESS on majority. "
        f"This does not support increasing majority reliance with age."
    )
else:
    response = 30
    explanation = (
        f"No significant relationship found between age and majority preference "
        f"(r={r:.3f}, p={p_corr:.4f}). Cultural differences {'are' if culture_sig else 'are not'} "
        f"significant (p={p_anova:.4f})."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nWrote conclusion.txt: response={response}")
print("explanation:", explanation)
