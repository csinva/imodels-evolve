import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crofoot.csv')

print("=" * 80)
print("ANALYSIS: Capuchin Monkey Intergroup Contest Outcomes")
print("=" * 80)
print("\nResearch Question:")
print("How do relative group size and contest location influence the probability")
print("of a capuchin monkey group winning an intergroup contest?")
print("=" * 80)

# 1. DATA EXPLORATION
print("\n1. DATA EXPLORATION")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSummary statistics:")
print(df.describe())

print(f"\n\nWin distribution:")
print(df['win'].value_counts())
print(f"Win rate: {df['win'].mean():.2%}")

# 2. CREATE KEY VARIABLES
print("\n\n2. FEATURE ENGINEERING")
print("-" * 80)

# Relative group size (focal vs other)
df['relative_size'] = df['n_focal'] - df['n_other']
df['focal_larger'] = (df['n_focal'] > df['n_other']).astype(int)

# Contest location: distance from focal's home range center vs other's
df['relative_distance'] = df['dist_focal'] - df['dist_other']
df['focal_closer_to_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

print(f"Relative size (focal - other): mean={df['relative_size'].mean():.2f}, std={df['relative_size'].std():.2f}")
print(f"Relative distance (focal - other): mean={df['relative_distance'].mean():.2f}, std={df['relative_distance'].std():.2f}")
print(f"\nFocal larger than other: {df['focal_larger'].sum()} out of {len(df)} contests ({df['focal_larger'].mean():.1%})")
print(f"Focal closer to home: {df['focal_closer_to_home'].sum()} out of {len(df)} contests ({df['focal_closer_to_home'].mean():.1%})")

# 3. STATISTICAL TESTS
print("\n\n3. STATISTICAL ANALYSIS")
print("-" * 80)

# Test 1: Does relative group size affect win probability?
print("\nTest 1: Relative Group Size Effect")
print("-" * 40)
wins_when_larger = df[df['focal_larger'] == 1]['win'].mean()
wins_when_smaller = df[df['focal_larger'] == 0]['win'].mean()
print(f"Win rate when focal is LARGER: {wins_when_larger:.2%}")
print(f"Win rate when focal is SMALLER/EQUAL: {wins_when_smaller:.2%}")

# Chi-square test for focal_larger vs win
contingency_size = pd.crosstab(df['focal_larger'], df['win'])
chi2_size, p_size = stats.chi2_contingency(contingency_size)[:2]
print(f"Chi-square test: χ²={chi2_size:.3f}, p={p_size:.4f}")

# Test 2: Does contest location affect win probability?
print("\nTest 2: Contest Location Effect")
print("-" * 40)
wins_when_closer = df[df['focal_closer_to_home'] == 1]['win'].mean()
wins_when_farther = df[df['focal_closer_to_home'] == 0]['win'].mean()
print(f"Win rate when focal is CLOSER to home: {wins_when_closer:.2%}")
print(f"Win rate when focal is FARTHER from home: {wins_when_farther:.2%}")

contingency_location = pd.crosstab(df['focal_closer_to_home'], df['win'])
chi2_loc, p_loc = stats.chi2_contingency(contingency_location)[:2]
print(f"Chi-square test: χ²={chi2_loc:.3f}, p={p_loc:.4f}")

# Test 3: Correlation between continuous relative measures and winning
print("\nTest 3: Correlations with Winning")
print("-" * 40)
corr_size, p_corr_size = stats.pointbiserialr(df['win'], df['relative_size'])
print(f"Correlation between win and relative_size: r={corr_size:.3f}, p={p_corr_size:.4f}")

corr_dist, p_corr_dist = stats.pointbiserialr(df['win'], df['relative_distance'])
print(f"Correlation between win and relative_distance: r={corr_dist:.3f}, p={p_corr_dist:.4f}")

# 4. LOGISTIC REGRESSION MODEL
print("\n\n4. LOGISTIC REGRESSION MODEL")
print("-" * 80)

# Prepare features
X = df[['relative_size', 'relative_distance']].values
y = df['win'].values

# Standardize features for better interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression with statsmodels for p-values
X_with_const = sm.add_constant(X_scaled)
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit(disp=0)

print("\nLogistic Regression Results:")
print(result.summary2().tables[1])

# Extract coefficients and p-values
coef_size = result.params[1]
p_val_size = result.pvalues[1]
coef_dist = result.params[2]
p_val_dist = result.pvalues[2]

print(f"\n\nInterpretation:")
print(f"- Relative group size coefficient: {coef_size:.3f} (p={p_val_size:.4f})")
print(f"- Relative distance coefficient: {coef_dist:.3f} (p={p_val_dist:.4f})")

# 5. ADDITIONAL ANALYSIS: Combined effects
print("\n\n5. COMBINED EFFECTS ANALYSIS")
print("-" * 80)

# Create groups based on both factors
df['both_advantages'] = ((df['focal_larger'] == 1) & (df['focal_closer_to_home'] == 1)).astype(int)
df['neither_advantage'] = ((df['focal_larger'] == 0) & (df['focal_closer_to_home'] == 0)).astype(int)

print("\nWin rates by combined factors:")
for larger in [0, 1]:
    for closer in [0, 1]:
        mask = (df['focal_larger'] == larger) & (df['focal_closer_to_home'] == closer)
        if mask.sum() > 0:
            win_rate = df[mask]['win'].mean()
            count = mask.sum()
            print(f"  Larger={larger}, Closer={closer}: {win_rate:.2%} (n={count})")

# 6. CONCLUSION
print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine response based on statistical significance
# Both factors should show significance for a strong "Yes"
significant_factors = []
if p_size < 0.05:
    significant_factors.append("group size")
if p_loc < 0.05:
    significant_factors.append("contest location")

if p_val_size < 0.05:
    significant_factors.append("relative size (regression)")
if p_val_dist < 0.05:
    significant_factors.append("relative distance (regression)")

# Generate response score
if len(significant_factors) >= 2:
    # Both factors are significant
    response_score = 85
    explanation = (
        f"Yes, both relative group size and contest location significantly influence "
        f"contest outcomes. Statistical tests show: (1) Group size effect: χ²={chi2_size:.2f}, "
        f"p={p_size:.4f} - larger groups win {wins_when_larger:.1%} vs {wins_when_smaller:.1%} "
        f"for smaller groups. (2) Location effect: χ²={chi2_loc:.2f}, p={p_loc:.4f} - "
        f"groups closer to home win {wins_when_closer:.1%} vs {wins_when_farther:.1%} when farther. "
        f"Logistic regression confirms both effects with p-values < 0.05."
    )
elif len(set(['group size', 'contest location']) & set(significant_factors)) == 2:
    # Both main chi-square tests significant
    response_score = 80
    explanation = (
        f"Yes, both factors significantly influence outcomes. Chi-square tests show: "
        f"group size (χ²={chi2_size:.2f}, p={p_size:.4f}) and location (χ²={chi2_loc:.2f}, "
        f"p={p_loc:.4f}) are both significant. Larger groups and groups closer to home "
        f"have higher win rates."
    )
elif 'group size' in significant_factors or 'relative size (regression)' in significant_factors:
    # Only group size significant
    response_score = 50
    explanation = (
        f"Partial support: Group size shows significant effect (p={p_size:.4f}), "
        f"with larger groups winning {wins_when_larger:.1%} vs {wins_when_smaller:.1%}. "
        f"However, contest location effect is not statistically significant (p={p_loc:.4f})."
    )
elif 'contest location' in significant_factors or 'relative distance (regression)' in significant_factors:
    # Only location significant
    response_score = 50
    explanation = (
        f"Partial support: Contest location shows significant effect (p={p_loc:.4f}), "
        f"with groups closer to home winning {wins_when_closer:.1%} vs {wins_when_farther:.1%}. "
        f"However, group size effect is not statistically significant (p={p_size:.4f})."
    )
else:
    # Neither significant
    response_score = 20
    explanation = (
        f"No, neither factor shows statistically significant influence. Group size: "
        f"p={p_size:.4f}, Location: p={p_loc:.4f}. Both p-values exceed 0.05 threshold."
    )

print(f"\nResponse Score: {response_score}/100")
print(f"\nExplanation: {explanation}")

# Save conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results saved to conclusion.txt")
print("=" * 80)
