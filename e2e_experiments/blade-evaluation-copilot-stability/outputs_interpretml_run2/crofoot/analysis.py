import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier

# Load data
df = pd.read_csv('crofoot.csv')

# Explore the data
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(df.head(10))
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Research question: How do relative group size and contest location influence 
# the probability of a capuchin monkey group winning an intergroup contest?

# Key variables:
# - win: outcome (1 if focal won, 0 if other won)
# - n_focal, n_other: group sizes
# - dist_focal, dist_other: distances from home range centers (location proxy)

# Create relative group size variable
df['relative_size'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']

# Create relative location variable (home advantage)
# Closer to home = smaller distance = advantage
df['relative_distance'] = df['dist_other'] - df['dist_focal']  # Positive if focal closer to home

print("\n" + "=" * 80)
print("KEY VARIABLES ANALYSIS")
print("=" * 80)
print("\nRelative size (n_focal - n_other):")
print(df['relative_size'].describe())
print("\nRelative distance (dist_other - dist_focal):")
print(df['relative_distance'].describe())
print("\nWin rate:", df['win'].mean())

# Split by win/loss to see differences
print("\n" + "=" * 80)
print("COMPARING WINNERS VS LOSERS")
print("=" * 80)
winners = df[df['win'] == 1]
losers = df[df['win'] == 0]

print(f"\nWins: {len(winners)}, Losses: {len(losers)}")
print("\nRelative size by outcome:")
print(f"  Winners: mean={winners['relative_size'].mean():.2f}, std={winners['relative_size'].std():.2f}")
print(f"  Losers: mean={losers['relative_size'].mean():.2f}, std={losers['relative_size'].std():.2f}")

print("\nRelative distance by outcome:")
print(f"  Winners: mean={winners['relative_distance'].mean():.2f}, std={winners['relative_distance'].std():.2f}")
print(f"  Losers: mean={losers['relative_distance'].mean():.2f}, std={losers['relative_distance'].std():.2f}")

# T-tests for significance
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

t_stat_size, p_val_size = stats.ttest_ind(winners['relative_size'], losers['relative_size'])
print(f"\nT-test for relative size: t={t_stat_size:.4f}, p={p_val_size:.4f}")
if p_val_size < 0.05:
    print("  *** SIGNIFICANT at p<0.05")
else:
    print("  Not significant at p<0.05")

t_stat_dist, p_val_dist = stats.ttest_ind(winners['relative_distance'], losers['relative_distance'])
print(f"\nT-test for relative distance: t={t_stat_dist:.4f}, p={p_val_dist:.4f}")
if p_val_dist < 0.05:
    print("  *** SIGNIFICANT at p<0.05")
else:
    print("  Not significant at p<0.05")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
corr_size = stats.pearsonr(df['relative_size'], df['win'])
print(f"\nCorrelation between relative size and win: r={corr_size[0]:.4f}, p={corr_size[1]:.4f}")

corr_dist = stats.pearsonr(df['relative_distance'], df['win'])
print(f"Correlation between relative distance and win: r={corr_dist[0]:.4f}, p={corr_dist[1]:.4f}")

# Logistic regression with statsmodels for p-values
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION (statsmodels)")
print("=" * 80)

X = df[['relative_size', 'relative_distance']]
y = df['win']

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['relative_size_scaled', 'relative_distance_scaled'])

# Add constant for intercept
X_with_const = sm.add_constant(X_scaled_df)

# Fit logistic regression
logit_model = sm.Logit(y, X_with_const)
logit_result = logit_model.fit(disp=0)

print("\nLogistic Regression Results:")
print(logit_result.summary())

# Extract coefficients and p-values
coef_size = logit_result.params['relative_size_scaled']
pval_size = logit_result.pvalues['relative_size_scaled']
coef_dist = logit_result.params['relative_distance_scaled']
pval_dist = logit_result.pvalues['relative_distance_scaled']

print(f"\nRelative size coefficient: {coef_size:.4f}, p-value: {pval_size:.4f}")
print(f"Relative distance coefficient: {coef_dist:.4f}, p-value: {pval_dist:.4f}")

# Interpretable Boosting Machine
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING CLASSIFIER")
print("=" * 80)

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X, y)

# Get feature importances
importances = ebm.term_importances()
print("\nFeature importances:")
for i, col in enumerate(X.columns):
    print(f"  {col}: {importances[i]:.4f}")

# Determine conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

significant_effects = []
non_significant_effects = []

if pval_size < 0.05:
    significant_effects.append("relative group size")
    print(f"✓ Relative group size has a SIGNIFICANT effect (p={pval_size:.4f})")
    print(f"  - Coefficient: {coef_size:.4f} (positive = larger groups win more)")
else:
    non_significant_effects.append("relative group size")
    print(f"✗ Relative group size does NOT have a significant effect (p={pval_size:.4f})")

if pval_dist < 0.05:
    significant_effects.append("contest location")
    print(f"✓ Contest location has a SIGNIFICANT effect (p={pval_dist:.4f})")
    print(f"  - Coefficient: {coef_dist:.4f} (positive = home advantage)")
else:
    non_significant_effects.append("contest location")
    print(f"✗ Contest location does NOT have a significant effect (p={pval_dist:.4f})")

# Determine response score
if len(significant_effects) == 2:
    # Both factors significant
    response_score = 90
    explanation = f"Both relative group size (p={pval_size:.4f}) and contest location (p={pval_dist:.4f}) have statistically significant effects on winning probability. Logistic regression shows that larger relative group size (coef={coef_size:.3f}) and being closer to home (coef={coef_dist:.3f}) both increase the probability of winning an intergroup contest."
elif len(significant_effects) == 1:
    # One factor significant
    if "relative group size" in significant_effects:
        response_score = 70
        explanation = f"Relative group size has a significant effect (p={pval_size:.4f}, coef={coef_size:.3f}) on winning probability, but contest location does not (p={pval_dist:.4f}). Larger groups have an advantage, but location appears less important."
    else:
        response_score = 70
        explanation = f"Contest location has a significant effect (p={pval_dist:.4f}, coef={coef_dist:.3f}) on winning probability, but relative group size does not (p={pval_size:.4f}). Being closer to home provides an advantage, but size differences appear less important."
else:
    # Neither significant
    response_score = 20
    explanation = f"Neither relative group size (p={pval_size:.4f}) nor contest location (p={pval_dist:.4f}) show statistically significant effects on winning probability in this dataset. The factors that influence intergroup contest outcomes remain unclear from this analysis."

print(f"\nResponse score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
