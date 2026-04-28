import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load the data
df = pd.read_csv('crofoot.csv')

print("=" * 80)
print("CAPUCHIN MONKEY INTERGROUP CONTEST ANALYSIS")
print("=" * 80)
print("\nResearch Question: How do relative group size and contest location")
print("influence the probability of winning an intergroup contest?")
print("\n" + "=" * 80)

# Display basic dataset info
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(df.describe())

# Create derived features for analysis
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Relative group size (focal vs other)
df['relative_size'] = df['n_focal'] - df['n_other']
df['relative_size_ratio'] = df['n_focal'] / df['n_other']

# Contest location: distance from home range center
# Negative means closer to focal's home, positive means closer to other's home
df['location_advantage'] = df['dist_other'] - df['dist_focal']

print("\nDerived features:")
print(f"- relative_size: difference in group sizes (n_focal - n_other)")
print(f"- relative_size_ratio: ratio of group sizes (n_focal / n_other)")
print(f"- location_advantage: distance advantage (dist_other - dist_focal)")
print(f"  Positive = closer to focal's home, Negative = closer to other's home")

print("\n" + "=" * 80)
print("EXPLORATORY ANALYSIS")
print("=" * 80)

# Win rate by relative size
print("\n1. Win rate by relative group size:")
print(df.groupby('relative_size')['win'].agg(['count', 'mean', 'std']))

# Win rate by location advantage (binned)
df['location_bin'] = pd.cut(df['location_advantage'], bins=3, labels=['Away', 'Neutral', 'Home'])
print("\n2. Win rate by location (Home = closer to focal's territory):")
print(df.groupby('location_bin')['win'].agg(['count', 'mean', 'std']))

# Correlations
print("\n3. Correlations with winning:")
correlations = df[['win', 'relative_size', 'location_advantage', 'n_focal', 'dist_focal']].corr()['win'].sort_values(ascending=False)
print(correlations)

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Effect of relative group size on winning
print("\n1. LOGISTIC REGRESSION: Relative Group Size Effect")
print("-" * 60)
X_size = df[['relative_size']]
y = df['win']
size_model = LogisticRegression()
size_model.fit(X_size, y)
print(f"Coefficient for relative_size: {size_model.coef_[0][0]:.4f}")
print(f"Interpretation: Each additional member gives {np.exp(size_model.coef_[0][0]):.3f}x odds of winning")

# Statsmodels for p-values
X_size_sm = sm.add_constant(X_size)
logit_size = sm.Logit(y, X_size_sm)
result_size = logit_size.fit(disp=0)
print("\nStatsmodels summary:")
print(result_size.summary2().tables[1])
print(f"\nP-value for relative_size: {result_size.pvalues['relative_size']:.6f}")
size_significant = result_size.pvalues['relative_size'] < 0.05
print(f"Statistically significant at α=0.05? {size_significant}")

# Test 2: Effect of location on winning
print("\n2. LOGISTIC REGRESSION: Contest Location Effect")
print("-" * 60)
X_loc = df[['location_advantage']]
loc_model = LogisticRegression()
loc_model.fit(X_loc, y)
print(f"Coefficient for location_advantage: {loc_model.coef_[0][0]:.6f}")

X_loc_sm = sm.add_constant(X_loc)
logit_loc = sm.Logit(y, X_loc_sm)
result_loc = logit_loc.fit(disp=0)
print("\nStatsmodels summary:")
print(result_loc.summary2().tables[1])
print(f"\nP-value for location_advantage: {result_loc.pvalues['location_advantage']:.6f}")
loc_significant = result_loc.pvalues['location_advantage'] < 0.05
print(f"Statistically significant at α=0.05? {loc_significant}")

# Test 3: Combined model with both factors
print("\n3. COMBINED LOGISTIC REGRESSION: Size + Location")
print("-" * 60)
X_combined = df[['relative_size', 'location_advantage']]
combined_model = LogisticRegression()
combined_model.fit(X_combined, y)

X_combined_sm = sm.add_constant(X_combined)
logit_combined = sm.Logit(y, X_combined_sm)
result_combined = logit_combined.fit(disp=0)
print("\nStatsmodels summary:")
print(result_combined.summary2().tables[1])
print(f"\nP-values:")
print(f"  relative_size: {result_combined.pvalues['relative_size']:.6f}")
print(f"  location_advantage: {result_combined.pvalues['location_advantage']:.6f}")
both_significant = (result_combined.pvalues['relative_size'] < 0.05 and 
                   result_combined.pvalues['location_advantage'] < 0.05)

# Test 4: Compare wins when focal is larger vs smaller
print("\n4. T-TEST: Win rates for larger vs smaller groups")
print("-" * 60)
wins_when_larger = df[df['relative_size'] > 0]['win']
wins_when_smaller = df[df['relative_size'] < 0]['win']
wins_when_equal = df[df['relative_size'] == 0]['win']

print(f"Win rate when focal is larger (n={len(wins_when_larger)}): {wins_when_larger.mean():.3f}")
print(f"Win rate when focal is smaller (n={len(wins_when_smaller)}): {wins_when_smaller.mean():.3f}")
print(f"Win rate when equal size (n={len(wins_when_equal)}): {wins_when_equal.mean():.3f}")

if len(wins_when_larger) > 0 and len(wins_when_smaller) > 0:
    t_stat, p_val = stats.ttest_ind(wins_when_larger, wins_when_smaller)
    print(f"\nT-test: larger vs smaller groups")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
    size_difference_significant = p_val < 0.05
    print(f"  Significant difference? {size_difference_significant}")

# Test 5: Home advantage
print("\n5. COMPARISON: Home vs Away contests")
print("-" * 60)
# Home = location_advantage > 0 (closer to focal's home)
home_contests = df[df['location_advantage'] > 0]['win']
away_contests = df[df['location_advantage'] < 0]['win']

print(f"Win rate at home (n={len(home_contests)}): {home_contests.mean():.3f}")
print(f"Win rate away (n={len(away_contests)}): {away_contests.mean():.3f}")

if len(home_contests) > 0 and len(away_contests) > 0:
    t_stat_loc, p_val_loc = stats.ttest_ind(home_contests, away_contests)
    print(f"\nT-test: home vs away")
    print(f"  t-statistic: {t_stat_loc:.4f}, p-value: {p_val_loc:.6f}")
    location_difference_significant = p_val_loc < 0.05
    print(f"  Significant difference? {location_difference_significant}")

print("\n" + "=" * 80)
print("FINAL INTERPRETATION")
print("=" * 80)

# Determine the answer
print("\nKey Findings:")
print(f"1. Relative Group Size:")
print(f"   - Coefficient: {result_combined.params['relative_size']:.4f}")
print(f"   - P-value: {result_combined.pvalues['relative_size']:.6f}")
print(f"   - Significant: {result_combined.pvalues['relative_size'] < 0.05}")

print(f"\n2. Contest Location:")
print(f"   - Coefficient: {result_combined.params['location_advantage']:.6f}")
print(f"   - P-value: {result_combined.pvalues['location_advantage']:.6f}")
print(f"   - Significant: {result_combined.pvalues['location_advantage'] < 0.05}")

# Determine response score
size_sig = result_combined.pvalues['relative_size'] < 0.05
loc_sig = result_combined.pvalues['location_advantage'] < 0.05

if size_sig and loc_sig:
    response = 90
    explanation = (
        f"Both relative group size (p={result_combined.pvalues['relative_size']:.4f}) and "
        f"contest location (p={result_combined.pvalues['location_advantage']:.4f}) significantly "
        f"influence winning probability. Larger groups have higher odds of winning, and groups "
        f"fighting closer to their home range have an advantage. The combined logistic regression "
        f"model shows both effects are statistically significant predictors."
    )
elif size_sig:
    response = 70
    explanation = (
        f"Relative group size significantly influences winning probability "
        f"(p={result_combined.pvalues['relative_size']:.4f}), with larger groups having "
        f"higher odds of winning. However, contest location does not show a statistically "
        f"significant effect (p={result_combined.pvalues['location_advantage']:.4f})."
    )
elif loc_sig:
    response = 70
    explanation = (
        f"Contest location significantly influences winning probability "
        f"(p={result_combined.pvalues['location_advantage']:.4f}), with home advantage "
        f"playing a role. However, relative group size does not show a statistically "
        f"significant effect (p={result_combined.pvalues['relative_size']:.4f})."
    )
else:
    response = 20
    explanation = (
        f"Neither relative group size (p={result_combined.pvalues['relative_size']:.4f}) nor "
        f"contest location (p={result_combined.pvalues['location_advantage']:.4f}) show "
        f"statistically significant effects on winning probability at the α=0.05 level. "
        f"With a small sample size (n={len(df)}), we cannot confidently conclude these "
        f"factors influence contest outcomes."
    )

print(f"\n{'='*80}")
print(f"CONCLUSION")
print(f"{'='*80}")
print(f"\nResponse Score: {response}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Analysis complete. Results written to conclusion.txt")
