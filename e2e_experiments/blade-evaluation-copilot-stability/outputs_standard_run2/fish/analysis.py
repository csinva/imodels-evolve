import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('fish.csv')

print("=" * 80)
print("RESEARCH QUESTION:")
print("How many fish on average do visitors catch per hour when fishing?")
print("What factors influence the number of fish caught?")
print("=" * 80)

# Explore the data
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Create fish per hour metric
df['fish_per_hour'] = df['fish_caught'] / df['hours']
# Handle division by zero or near-zero hours
df['fish_per_hour'] = df['fish_per_hour'].replace([np.inf, -np.inf], np.nan)

print("\n" + "=" * 80)
print("FISH PER HOUR ANALYSIS")
print("=" * 80)

# Remove rows where hours is too small or fish_per_hour is invalid
valid_df = df[df['hours'] > 0.01].copy()
valid_df['fish_per_hour'] = valid_df['fish_caught'] / valid_df['hours']

# Cap extreme values (remove outliers)
q99 = valid_df['fish_per_hour'].quantile(0.99)
valid_df_clean = valid_df[valid_df['fish_per_hour'] <= q99].copy()

print(f"\nValid observations (hours > 0.01): {len(valid_df)}")
print(f"After removing extreme outliers (> 99th percentile): {len(valid_df_clean)}")

print("\nFish per hour statistics:")
print(valid_df_clean['fish_per_hour'].describe())

mean_fish_per_hour = valid_df_clean['fish_per_hour'].mean()
median_fish_per_hour = valid_df_clean['fish_per_hour'].median()
std_fish_per_hour = valid_df_clean['fish_per_hour'].std()

print(f"\nMean fish caught per hour: {mean_fish_per_hour:.3f}")
print(f"Median fish caught per hour: {median_fish_per_hour:.3f}")
print(f"Std dev: {std_fish_per_hour:.3f}")

print("\n" + "=" * 80)
print("FACTOR ANALYSIS - What influences fish caught?")
print("=" * 80)

# Correlation analysis
print("\nCorrelation of fish_caught with other variables:")
correlations = df[['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()['fish_caught'].sort_values(ascending=False)
print(correlations)

# Statistical tests for categorical variables
print("\n" + "-" * 40)
print("T-test: Livebait vs No Livebait")
print("-" * 40)
livebait_yes = df[df['livebait'] == 1]['fish_caught']
livebait_no = df[df['livebait'] == 0]['fish_caught']
t_stat_livebait, p_val_livebait = stats.ttest_ind(livebait_yes, livebait_no)
print(f"Livebait=1 mean: {livebait_yes.mean():.3f}, Livebait=0 mean: {livebait_no.mean():.3f}")
print(f"T-statistic: {t_stat_livebait:.3f}, p-value: {p_val_livebait:.6f}")
print(f"Significant: {p_val_livebait < 0.05}")

print("\n" + "-" * 40)
print("T-test: Camper vs No Camper")
print("-" * 40)
camper_yes = df[df['camper'] == 1]['fish_caught']
camper_no = df[df['camper'] == 0]['fish_caught']
t_stat_camper, p_val_camper = stats.ttest_ind(camper_yes, camper_no)
print(f"Camper=1 mean: {camper_yes.mean():.3f}, Camper=0 mean: {camper_no.mean():.3f}")
print(f"T-statistic: {t_stat_camper:.3f}, p-value: {p_val_camper:.6f}")
print(f"Significant: {p_val_camper < 0.05}")

# Linear regression with statsmodels for p-values
print("\n" + "=" * 80)
print("LINEAR REGRESSION ANALYSIS (with p-values)")
print("=" * 80)

X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
y = df['fish_caught']
X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const).fit()
print(model.summary())

print("\n" + "-" * 40)
print("Key findings from regression:")
print("-" * 40)
print(f"R-squared: {model.rsquared:.4f}")
print("\nCoefficients and significance:")
for var in X.columns:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "not sig"
    print(f"  {var:12s}: coef={coef:8.4f}, p={pval:.6f} {sig}")

# Interpretable model with sklearn
print("\n" + "=" * 80)
print("DECISION TREE for Interpretability")
print("=" * 80)

dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
dt.fit(X, y)

print("\nFeature importances:")
for feat, imp in zip(X.columns, dt.feature_importances_):
    print(f"  {feat:12s}: {imp:.4f}")

# Cross-validation score
cv_scores = cross_val_score(dt, X, y, cv=5, scoring='r2')
print(f"\nDecision Tree CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Linear regression with sklearn
lr = LinearRegression()
lr.fit(X, y)
lr_cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"Linear Regression CV R² score: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Determine the response based on statistical evidence
significant_factors = []
explanation_parts = []

# Check each factor
if p_val_livebait < 0.05:
    significant_factors.append('livebait')
    explanation_parts.append(f"livebait shows significant effect (p={p_val_livebait:.4f}, mean diff={livebait_yes.mean() - livebait_no.mean():.2f} fish)")

if p_val_camper < 0.05:
    significant_factors.append('camper')
    explanation_parts.append(f"camper shows significant effect (p={p_val_camper:.4f})")

# Check regression coefficients
for var in ['persons', 'child', 'hours']:
    if model.pvalues[var] < 0.05:
        significant_factors.append(var)
        explanation_parts.append(f"{var} is significant (p={model.pvalues[var]:.4f}, coef={model.params[var]:.3f})")

# The primary question is about estimating fish per hour
print(f"\nAverage fish caught per hour: {mean_fish_per_hour:.3f} fish/hour")
print(f"Median fish caught per hour: {median_fish_per_hour:.3f} fish/hour")

print(f"\nSignificant factors influencing fish caught: {significant_factors}")
print(f"Number of significant factors: {len(significant_factors)}")

# Build explanation
if len(significant_factors) > 0:
    base_explanation = f"Visitors catch an average of {mean_fish_per_hour:.2f} fish per hour (median: {median_fish_per_hour:.2f}). "
    base_explanation += f"Analysis identified {len(significant_factors)} significant factors: "
    base_explanation += "; ".join(explanation_parts) + ". "
    base_explanation += f"The regression model has R²={model.rsquared:.3f}, indicating that these factors explain {model.rsquared*100:.1f}% of variance in catch rates."
    
    # Score based on strength of findings
    # Strong evidence (multiple significant factors, decent R²) -> high score
    if model.rsquared > 0.15 and len(significant_factors) >= 2:
        response_score = 85
    elif model.rsquared > 0.10 or len(significant_factors) >= 3:
        response_score = 75
    elif len(significant_factors) >= 1:
        response_score = 70
    else:
        response_score = 60
else:
    base_explanation = f"Visitors catch an average of {mean_fish_per_hour:.2f} fish per hour (median: {median_fish_per_hour:.2f}). "
    base_explanation += "However, none of the analyzed factors (livebait, camper, persons, child) show statistically significant relationships with catch rates. "
    base_explanation += f"The regression model has low explanatory power (R²={model.rsquared:.3f})."
    response_score = 50

print(f"\nResponse score: {response_score}/100")
print(f"Explanation: {base_explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": base_explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
