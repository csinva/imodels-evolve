import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor
import json
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('fish.csv')

print("=" * 80)
print("FISH CATCH ANALYSIS")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Total records: {len(df)}")

# Basic exploration
print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("CORRELATIONS WITH FISH CAUGHT")
print("=" * 80)
correlations = df.corr()['fish_caught'].sort_values(ascending=False)
print(correlations)

# Create fish per hour rate (handling zero hours)
df['fish_per_hour'] = np.where(df['hours'] > 0, df['fish_caught'] / df['hours'], 0)

print("\n" + "=" * 80)
print("FISH PER HOUR STATISTICS")
print("=" * 80)
# Only calculate for visits where hours > 0
valid_hours = df[df['hours'] > 0]['fish_per_hour']
print(f"Mean fish per hour: {valid_hours.mean():.3f}")
print(f"Median fish per hour: {valid_hours.median():.3f}")
print(f"Std fish per hour: {valid_hours.std():.3f}")
print(f"Min fish per hour: {valid_hours.min():.3f}")
print(f"Max fish per hour: {valid_hours.max():.3f}")

# Statistical tests for each factor
print("\n" + "=" * 80)
print("STATISTICAL TESTS - FACTORS INFLUENCING FISH CAUGHT")
print("=" * 80)

# Test if livebait matters
livebait_yes = df[df['livebait'] == 1]['fish_caught']
livebait_no = df[df['livebait'] == 0]['fish_caught']
t_stat_bait, p_val_bait = stats.ttest_ind(livebait_yes, livebait_no)
print(f"\nLivebait effect:")
print(f"  With livebait: mean={livebait_yes.mean():.3f}, n={len(livebait_yes)}")
print(f"  Without livebait: mean={livebait_no.mean():.3f}, n={len(livebait_no)}")
print(f"  T-test: t={t_stat_bait:.3f}, p={p_val_bait:.4f}")

# Test if camper matters
camper_yes = df[df['camper'] == 1]['fish_caught']
camper_no = df[df['camper'] == 0]['fish_caught']
t_stat_camper, p_val_camper = stats.ttest_ind(camper_yes, camper_no)
print(f"\nCamper effect:")
print(f"  With camper: mean={camper_yes.mean():.3f}, n={len(camper_yes)}")
print(f"  Without camper: mean={camper_no.mean():.3f}, n={len(camper_no)}")
print(f"  T-test: t={t_stat_camper:.3f}, p={p_val_camper:.4f}")

# Correlation tests for continuous variables
corr_hours, p_hours = stats.pearsonr(df['hours'], df['fish_caught'])
print(f"\nHours correlation with fish caught:")
print(f"  Correlation: r={corr_hours:.3f}, p={p_hours:.4f}")

corr_persons, p_persons = stats.pearsonr(df['persons'], df['fish_caught'])
print(f"\nPersons correlation with fish caught:")
print(f"  Correlation: r={corr_persons:.3f}, p={p_persons:.4f}")

corr_child, p_child = stats.pearsonr(df['child'], df['fish_caught'])
print(f"\nChild correlation with fish caught:")
print(f"  Correlation: r={corr_child:.3f}, p={p_child:.4f}")

# Linear Regression with statsmodels for p-values
print("\n" + "=" * 80)
print("LINEAR REGRESSION - ALL FACTORS")
print("=" * 80)
X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
y = df['fish_caught']
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(model_sm.summary())

# Interpretable models
print("\n" + "=" * 80)
print("INTERPRETABLE MODELS")
print("=" * 80)

# Linear regression coefficients
lr = LinearRegression()
lr.fit(X, y)
print("\nLinear Regression Coefficients:")
for feat, coef in zip(X.columns, lr.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {lr.intercept_:.4f}")
print(f"  R²: {lr.score(X, y):.4f}")

# Decision Tree for feature importance
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(X, y)
print("\nDecision Tree Feature Importances:")
for feat, imp in zip(X.columns, dt.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# RuleFit for interpretable rules
print("\nRuleFit Model (Rule-based interpretation):")
rulefit = RuleFitRegressor(max_rules=10, random_state=42)
rulefit.fit(X, y)
print(f"  R²: {rulefit.score(X, y):.4f}")

# Try FIGS (Fast Interpretable Greedy-tree Sums)
print("\nFIGS Model (Interpretable tree-based):")
figs = FIGSRegressor(max_rules=10, random_state=42)
figs.fit(X, y)
print(f"  R²: {figs.score(X, y):.4f}")

# Fish per hour regression
print("\n" + "=" * 80)
print("FISH PER HOUR RATE ESTIMATION")
print("=" * 80)
df_valid_hours = df[df['hours'] > 0].copy()
X_rate = df_valid_hours[['livebait', 'camper', 'persons', 'child']]
y_rate = df_valid_hours['fish_per_hour']

# Use log transformation for rate (often better for count rates)
y_rate_log = np.log1p(y_rate)

X_rate_const = sm.add_constant(X_rate)
model_rate = sm.OLS(y_rate_log, X_rate_const).fit()
print("Log-transformed Fish Per Hour Regression:")
print(model_rate.summary())

# Analysis and conclusion
print("\n" + "=" * 80)
print("INTERPRETATION AND CONCLUSION")
print("=" * 80)

significant_factors = []
factor_effects = {}

# Check which factors are significant (p < 0.05)
if p_val_bait < 0.05:
    significant_factors.append('livebait')
    factor_effects['livebait'] = livebait_yes.mean() - livebait_no.mean()

if p_val_camper < 0.05:
    significant_factors.append('camper')
    factor_effects['camper'] = camper_yes.mean() - camper_no.mean()

if p_hours < 0.05:
    significant_factors.append('hours')
    factor_effects['hours'] = corr_hours

if p_persons < 0.05:
    significant_factors.append('persons')
    factor_effects['persons'] = corr_persons

if p_child < 0.05:
    significant_factors.append('child')
    factor_effects['child'] = corr_child

print(f"\nSignificant factors (p < 0.05): {significant_factors}")
print(f"\nFactor effects: {factor_effects}")

# Check R² from the full regression model
r_squared = model_sm.rsquared
print(f"\nFull model R²: {r_squared:.4f}")

# Calculate average fish per hour for the answer
avg_fish_per_hour = valid_hours.mean()
print(f"\nAverage fish caught per hour: {avg_fish_per_hour:.3f}")

# Determine response score (0-100)
# The question asks about factors influencing fish caught and estimating rate per hour
# We found:
# 1. Hours is HIGHLY significant (p < 0.001) with strong correlation
# 2. Multiple factors significantly influence catch
# 3. We can estimate rate per hour successfully
# 4. R² shows decent predictive power

# Build explanation
explanation_parts = []

if 'hours' in significant_factors:
    explanation_parts.append(f"Hours spent fishing is highly significant (p={p_hours:.4f}, r={corr_hours:.3f})")

if 'livebait' in significant_factors:
    explanation_parts.append(f"Using livebait significantly increases catch (p={p_val_bait:.4f}, mean difference={factor_effects['livebait']:.2f} fish)")

if 'camper' in significant_factors:
    explanation_parts.append(f"Having a camper affects catch (p={p_val_camper:.4f})")

explanation_parts.append(f"Average fish caught per hour: {avg_fish_per_hour:.3f}")
explanation_parts.append(f"Regression model R²={r_squared:.3f} shows factors explain {r_squared*100:.1f}% of variance")

# Score determination:
# - We can successfully estimate rate (avg fish per hour)
# - We identified significant factors with statistical support
# - R² > 0.5 indicates good explanatory power
# Strong YES: 75-100, since we have clear significant factors and can estimate rate

if len(significant_factors) >= 2 and r_squared > 0.4:
    response_score = 85  # Strong evidence with multiple significant factors
elif len(significant_factors) >= 1 and r_squared > 0.3:
    response_score = 75  # Good evidence
elif len(significant_factors) >= 1:
    response_score = 65  # Moderate evidence
else:
    response_score = 40  # Weak evidence

explanation = " ".join(explanation_parts)

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
