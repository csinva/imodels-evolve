import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from imodels import RuleFitClassifier, FIGSClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crofoot.csv')

print("="*80)
print("CAPUCHIN MONKEY INTERGROUP CONTEST ANALYSIS")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Basic statistics
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(df.describe())

# Research question: How do relative group size and contest location influence winning?

# Create derived features
# 1. Relative group size (focal vs other)
df['size_diff'] = df['n_focal'] - df['n_other']  # Positive = focal larger
df['size_ratio'] = df['n_focal'] / df['n_other']

# 2. Contest location (home advantage)
# Distance from home: closer to focal's home = lower dist_focal, higher dist_other
df['location_advantage'] = df['dist_other'] - df['dist_focal']  # Positive = closer to focal's home

# 3. Other derived features
df['male_diff'] = df['m_focal'] - df['m_other']
df['female_diff'] = df['f_focal'] - df['f_other']

print("\n" + "="*80)
print("DERIVED FEATURES")
print("="*80)
print(f"\nSize difference (n_focal - n_other):")
print(df['size_diff'].describe())
print(f"\nLocation advantage (dist_other - dist_focal):")
print(df['location_advantage'].describe())

# Win rate by size difference
print("\n" + "="*80)
print("WIN RATE BY GROUP SIZE DIFFERENCE")
print("="*80)
size_groups = pd.cut(df['size_diff'], bins=[-10, -2, 0, 2, 10], labels=['Much smaller', 'Smaller', 'Larger', 'Much larger'])
win_by_size = df.groupby(size_groups)['win'].agg(['mean', 'count'])
print(win_by_size)

# Win rate by location
print("\n" + "="*80)
print("WIN RATE BY CONTEST LOCATION")
print("="*80)
loc_groups = pd.cut(df['location_advantage'], bins=[-1000, -100, 0, 100, 1000], labels=['Away', 'Neutral-Away', 'Neutral-Home', 'Home'])
win_by_loc = df.groupby(loc_groups)['win'].agg(['mean', 'count'])
print(win_by_loc)

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS")
print("="*80)

# 1. Correlation tests
print("\n1. CORRELATION TESTS:")
correlations = {
    'size_diff': stats.pearsonr(df['size_diff'], df['win']),
    'location_advantage': stats.pearsonr(df['location_advantage'], df['win']),
    'size_ratio': stats.pearsonr(df['size_ratio'], df['win']),
    'male_diff': stats.pearsonr(df['male_diff'], df['win']),
    'female_diff': stats.pearsonr(df['female_diff'], df['win'])
}

for var, (corr, pval) in correlations.items():
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"   {var}: r={corr:.3f}, p={pval:.4f} {sig}")

# 2. T-tests: Do winners have different characteristics?
print("\n2. T-TESTS (Winners vs Losers):")
winners = df[df['win'] == 1]
losers = df[df['win'] == 0]

for var in ['size_diff', 'location_advantage', 'n_focal', 'dist_focal']:
    t_stat, p_val = stats.ttest_ind(winners[var], losers[var])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"   {var}: t={t_stat:.3f}, p={p_val:.4f} {sig}")
    print(f"      Winners mean: {winners[var].mean():.2f}, Losers mean: {losers[var].mean():.2f}")

# Logistic regression with statsmodels for p-values
print("\n" + "="*80)
print("LOGISTIC REGRESSION (statsmodels)")
print("="*80)

# Use only the main variables to avoid multicollinearity
X_vars = ['size_diff', 'location_advantage']
X = df[X_vars]
X = sm.add_constant(X)
y = df['win']

try:
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=0)
    print(result.summary())
    size_coef_pval = result.pvalues['size_diff']
    loc_coef_pval = result.pvalues['location_advantage']
except Exception as e:
    print(f"Logistic regression error: {e}")
    print("Using correlation p-values instead")
    size_coef_pval = correlations['size_diff'][1]
    loc_coef_pval = correlations['location_advantage'][1]

# Interpretable ML models
print("\n" + "="*80)
print("INTERPRETABLE MODEL: RULEFIT")
print("="*80)

X_ml = df[['size_diff', 'location_advantage', 'male_diff', 'female_diff', 
           'n_focal', 'n_other', 'dist_focal', 'dist_other']]
y_ml = df['win']

# Standardize for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ml)

try:
    rf_model = RuleFitClassifier(max_rules=10, random_state=42)
    rf_model.fit(X_scaled, y_ml)
    print("\nRuleFit Rules:")
    if hasattr(rf_model, 'rules_'):
        for i, rule in enumerate(rf_model.rules_[:5]):  # Top 5 rules
            print(f"Rule {i+1}: {rule}")
except Exception as e:
    print(f"RuleFit error: {e}")

print("\n" + "="*80)
print("INTERPRETABLE MODEL: FIGS (Fast Interpretable Greedy-tree Sums)")
print("="*80)

try:
    figs_model = FIGSClassifier(max_rules=5)
    figs_model.fit(X_ml, y_ml)
    print(f"\nFIGS Score: {figs_model.score(X_ml, y_ml):.3f}")
    print("\nFIGS model structure:")
    print(figs_model)
except Exception as e:
    print(f"FIGS error: {e}")

# Simple logistic regression for coefficient interpretation
print("\n" + "="*80)
print("SCIKIT-LEARN LOGISTIC REGRESSION (Coefficients)")
print("="*80)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_scaled, y_ml)

feature_importance = pd.DataFrame({
    'feature': X_ml.columns,
    'coefficient': lr.coef_[0],
    'abs_coef': np.abs(lr.coef_[0])
}).sort_values('abs_coef', ascending=False)

print(feature_importance)

# Summary analysis
print("\n" + "="*80)
print("SYNTHESIS AND CONCLUSION")
print("="*80)

# Determine significance and effect direction
size_corr, size_pval = correlations['size_diff']
loc_corr, loc_pval = correlations['location_advantage']

size_sig = size_pval < 0.05
loc_sig = loc_pval < 0.05

print(f"\n1. RELATIVE GROUP SIZE:")
print(f"   - Correlation with winning: r={size_corr:.3f}, p={size_pval:.4f}")
print(f"   - Statistically significant: {size_sig}")
print(f"   - Effect direction: {'Larger groups win more' if size_corr > 0 else 'Smaller groups win more'}")
print(f"   - Winners avg size diff: {winners['size_diff'].mean():.2f}")
print(f"   - Losers avg size diff: {losers['size_diff'].mean():.2f}")

print(f"\n2. CONTEST LOCATION:")
print(f"   - Correlation with winning: r={loc_corr:.3f}, p={loc_pval:.4f}")
print(f"   - Statistically significant: {loc_sig}")
print(f"   - Effect direction: {'Home advantage exists' if loc_corr > 0 else 'No clear home advantage'}")
print(f"   - Winners avg location advantage: {winners['location_advantage'].mean():.2f}")
print(f"   - Losers avg location advantage: {losers['location_advantage'].mean():.2f}")

print(f"\n3. MULTIVARIATE LOGISTIC REGRESSION:")
print(f"   - Size difference coefficient p-value: {size_coef_pval:.4f}")
print(f"   - Location advantage coefficient p-value: {loc_coef_pval:.4f}")

# Overall assessment
both_significant = size_sig and loc_sig
one_significant = size_sig or loc_sig
neither_significant = not size_sig and not loc_sig

if both_significant:
    response_score = 85
    explanation = f"Both relative group size (r={size_corr:.3f}, p={size_pval:.4f}) and contest location (r={loc_corr:.3f}, p={loc_pval:.4f}) show statistically significant relationships with winning probability. Larger groups have an advantage, and contests closer to home territory also increase win probability."
elif size_sig and not loc_sig:
    response_score = 65
    explanation = f"Relative group size shows a significant effect (r={size_corr:.3f}, p={size_pval:.4f}), with larger groups winning more often. However, contest location does not show a statistically significant independent effect (p={loc_pval:.4f})."
elif loc_sig and not size_sig:
    response_score = 65
    explanation = f"Contest location shows a significant effect (r={loc_corr:.3f}, p={loc_pval:.4f}), with home advantage influencing outcomes. However, relative group size does not show a statistically significant independent effect (p={size_pval:.4f})."
else:
    response_score = 25
    explanation = f"Neither relative group size (p={size_pval:.4f}) nor contest location (p={loc_pval:.4f}) show statistically significant relationships with winning probability at the 0.05 alpha level. The data does not provide strong evidence for these factors influencing contest outcomes."

print(f"\n4. FINAL ASSESSMENT:")
print(f"   Response Score: {response_score}/100")
print(f"   Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results saved to conclusion.txt")
print("="*80)
