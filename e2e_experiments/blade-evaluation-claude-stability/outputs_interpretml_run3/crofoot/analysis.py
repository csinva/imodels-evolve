import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crofoot.csv')
print("Shape:", df.shape)
print(df.describe())

# Create key derived features
df['rel_size'] = df['n_focal'] / df['n_other']           # relative group size
df['loc_focal_adv'] = df['dist_other'] - df['dist_focal'] # positive = focal closer to home (home advantage)
df['size_diff'] = df['n_focal'] - df['n_other']

print("\n--- Correlation with win ---")
corr_cols = ['rel_size', 'size_diff', 'loc_focal_adv', 'dist_focal', 'dist_other', 'n_focal', 'n_other']
for col in corr_cols:
    r, p = stats.pointbiserialr(df[col], df['win'])
    print(f"  {col}: r={r:.3f}, p={p:.4f}")

# Logistic regression with statsmodels for p-values and CIs
print("\n--- Logistic Regression (statsmodels) ---")
X = df[['rel_size', 'loc_focal_adv']].copy()
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X).fit(disp=False)
print(model.summary())

# Individual tests
wins_larger = df[df['n_focal'] > df['n_other']]['win']
wins_smaller = df[df['n_focal'] < df['n_other']]['win']
wins_equal = df[df['n_focal'] == df['n_other']]['win']
print(f"\nWin rate when focal LARGER: {wins_larger.mean():.3f} (n={len(wins_larger)})")
print(f"Win rate when focal SMALLER: {wins_smaller.mean():.3f} (n={len(wins_smaller)})")
print(f"Win rate when equal size: {wins_equal.mean():.3f} (n={len(wins_equal)})")

# Home advantage: split by whether focal is closer to its home range
df['focal_home_adv'] = df['dist_focal'] < df['dist_other']
wins_home = df[df['focal_home_adv']]['win']
wins_away = df[~df['focal_home_adv']]['win']
print(f"\nWin rate when focal has home advantage (closer to home): {wins_home.mean():.3f} (n={len(wins_home)})")
print(f"Win rate when focal at away disadvantage: {wins_away.mean():.3f} (n={len(wins_away)})")
t_stat, p_home = stats.ttest_ind(wins_home, wins_away)
print(f"t-test p-value for home advantage: {p_home:.4f}")

# Full logistic regression with more features
print("\n--- Extended Logistic Regression ---")
X2 = df[['rel_size', 'loc_focal_adv', 'dist_focal', 'dist_other']].copy()
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2).fit(disp=False)
print(model2.summary())

# EBM for interpretable model
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    features = ['rel_size', 'loc_focal_adv', 'dist_focal', 'dist_other', 'size_diff']
    X_ebm = df[features]
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X_ebm, y)
    print("\n--- EBM Feature Importances ---")
    for name, imp in zip(features, ebm.term_importances()):
        print(f"  {name}: {imp:.4f}")
except Exception as e:
    print(f"EBM skipped: {e}")

# Summarize results for conclusion
p_relsize = model.pvalues['rel_size']
p_loc = model.pvalues['loc_focal_adv']
coef_relsize = model.params['rel_size']
coef_loc = model.params['loc_focal_adv']

print(f"\n=== SUMMARY ===")
print(f"Relative group size: coef={coef_relsize:.3f}, p={p_relsize:.4f}")
print(f"Location advantage: coef={coef_loc:.3f}, p={p_loc:.4f}")

# Both are significant -> strong Yes (high score)
# Determine response score
both_sig = (p_relsize < 0.05) and (p_loc < 0.05)
size_sig = p_relsize < 0.05
loc_sig = p_loc < 0.05

if both_sig:
    response = 90
    explanation = (
        f"Both relative group size and contest location significantly influence win probability. "
        f"Logistic regression: rel_size coef={coef_relsize:.3f} (p={p_relsize:.4f}), "
        f"loc_advantage coef={coef_loc:.3f} (p={p_loc:.4f}). "
        f"Larger focal groups win more often ({wins_larger.mean():.2f} vs {wins_smaller.mean():.2f} for smaller groups). "
        f"Focal groups with home advantage (closer to their home range) win more ({wins_home.mean():.2f} vs {wins_away.mean():.2f}). "
        f"Both factors jointly and significantly predict contest outcomes."
    )
elif size_sig and not loc_sig:
    response = 65
    explanation = (
        f"Relative group size significantly influences win probability (p={p_relsize:.4f}), "
        f"but contest location (home advantage) is not significant (p={p_loc:.4f}). "
        f"Partial support for the research question."
    )
elif not size_sig and loc_sig:
    response = 65
    explanation = (
        f"Contest location significantly influences win probability (p={p_loc:.4f}), "
        f"but relative group size is not significant (p={p_relsize:.4f}). "
        f"Partial support for the research question."
    )
else:
    response = 30
    explanation = (
        f"Neither relative group size (p={p_relsize:.4f}) nor contest location (p={p_loc:.4f}) "
        f"show statistically significant effects in this logistic regression model."
    )

result = {"response": response, "explanation": explanation}
print(f"\nConclusion: {json.dumps(result, indent=2)}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
