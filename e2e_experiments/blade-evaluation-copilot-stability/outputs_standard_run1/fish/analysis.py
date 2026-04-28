import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('fish.csv')

print("=" * 80)
print("FISH CATCH ANALYSIS")
print("=" * 80)
print("\n1. DATASET OVERVIEW")
print(f"Number of observations: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head(10))

print("\n2. SUMMARY STATISTICS")
print(df.describe())

print("\n3. DATA EXPLORATION")
print(f"\nFish caught - Mean: {df['fish_caught'].mean():.2f}, Median: {df['fish_caught'].median():.2f}")
print(f"Hours spent - Mean: {df['hours'].mean():.2f}, Median: {df['hours'].median():.2f}")

# Calculate fish per hour for non-zero hours
df['fish_per_hour'] = df['fish_caught'] / df['hours']
# Remove infinite values (where hours = 0 or very close to 0)
df_valid = df[df['hours'] > 0.01].copy()

print(f"\nFish per hour (excluding very short visits < 0.01 hours):")
print(f"  Mean: {df_valid['fish_per_hour'].mean():.3f}")
print(f"  Median: {df_valid['fish_per_hour'].median():.3f}")
print(f"  Std Dev: {df_valid['fish_per_hour'].std():.3f}")
print(f"  Min: {df_valid['fish_per_hour'].min():.3f}")
print(f"  Max: {df_valid['fish_per_hour'].max():.3f}")

print("\n4. CORRELATION ANALYSIS")
correlation_matrix = df[['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()
print("\nCorrelation with fish_caught:")
print(correlation_matrix['fish_caught'].sort_values(ascending=False))

print("\n5. FACTOR ANALYSIS - Impact on Fish Caught")

# Livebait analysis
livebait_yes = df[df['livebait'] == 1]['fish_caught']
livebait_no = df[df['livebait'] == 0]['fish_caught']
t_stat_bait, p_val_bait = stats.ttest_ind(livebait_yes, livebait_no)
print(f"\nLivebait effect:")
print(f"  With livebait: mean={livebait_yes.mean():.2f}, median={livebait_yes.median():.2f}")
print(f"  Without livebait: mean={livebait_no.mean():.2f}, median={livebait_no.median():.2f}")
print(f"  T-test: t={t_stat_bait:.3f}, p={p_val_bait:.4f}")

# Camper analysis
camper_yes = df[df['camper'] == 1]['fish_caught']
camper_no = df[df['camper'] == 0]['fish_caught']
t_stat_camp, p_val_camp = stats.ttest_ind(camper_yes, camper_no)
print(f"\nCamper effect:")
print(f"  With camper: mean={camper_yes.mean():.2f}, median={camper_yes.median():.2f}")
print(f"  Without camper: mean={camper_no.mean():.2f}, median={camper_no.median():.2f}")
print(f"  T-test: t={t_stat_camp:.3f}, p={p_val_camp:.4f}")

# Correlation with hours
corr_hours, p_hours = stats.pearsonr(df['fish_caught'], df['hours'])
print(f"\nHours correlation with fish caught:")
print(f"  Pearson r={corr_hours:.3f}, p={p_hours:.4f}")

print("\n6. REGRESSION ANALYSIS - Predicting Fish Caught")

# Prepare features
X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
y = df['fish_caught']

# Linear Regression with statsmodels for p-values
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print("\nOLS Regression Results:")
print(model_sm.summary())

# Interpretable model - Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
print("\n7. LINEAR REGRESSION COEFFICIENTS")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {lr_model.intercept_:.4f}")

# Calculate R-squared
from sklearn.metrics import r2_score
y_pred = lr_model.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nR-squared: {r2:.4f}")

print("\n8. FISH PER HOUR RATE ESTIMATION")

# Multiple approaches to estimate fish per hour
approach1_mean = df_valid['fish_per_hour'].mean()
approach1_median = df_valid['fish_per_hour'].median()

# Regression-based approach: coefficient of hours in the model
hours_coefficient = lr_model.coef_[-1]  # Last coefficient is hours

# Alternative: use only hours as predictor
X_hours = df[['hours']]
y_fish = df['fish_caught']
lr_hours = LinearRegression()
lr_hours.fit(X_hours, y_fish)
approach2_rate = lr_hours.coef_[0]

print(f"\nApproach 1 - Direct calculation (fish/hours):")
print(f"  Mean fish per hour: {approach1_mean:.3f}")
print(f"  Median fish per hour: {approach1_median:.3f}")

print(f"\nApproach 2 - Simple linear regression (fish ~ hours):")
print(f"  Fish per hour rate: {approach2_rate:.3f}")
print(f"  Intercept: {lr_hours.intercept_:.3f}")

print(f"\nApproach 3 - Multiple regression coefficient for hours:")
print(f"  Fish per hour rate (controlling for other factors): {hours_coefficient:.3f}")

print("\n9. KEY FINDINGS")

# Statistical significance threshold
alpha = 0.05

findings = []
findings.append(f"1. Average fish catch rate (direct calculation): {approach1_mean:.3f} fish/hour (median: {approach1_median:.3f})")
findings.append(f"2. Hours spent fishing has {'SIGNIFICANT' if p_hours < alpha else 'NO SIGNIFICANT'} correlation with fish caught (r={corr_hours:.3f}, p={p_hours:.4f})")
findings.append(f"3. Using livebait {'SIGNIFICANTLY' if p_val_bait < alpha else 'DOES NOT SIGNIFICANTLY'} increases fish caught (p={p_val_bait:.4f})")
findings.append(f"4. Having a camper {'SIGNIFICANTLY' if p_val_camp < alpha else 'DOES NOT SIGNIFICANTLY'} affects fish caught (p={p_val_camp:.4f})")

for finding in findings:
    print(f"  {finding}")

print("\n10. CONCLUSION")

# Calculate response score (0-100 scale)
# The question asks about average fish per hour and factors that influence it
# We have strong evidence:
# - We can calculate the average rate: ~1-2 fish per hour
# - Hours has significant positive correlation
# - Livebait has significant effect
# - We have a working model to estimate the rate

# Evidence strength:
evidence_score = 0

# Can we estimate the rate? YES - we have multiple estimates
evidence_score += 30

# Do we have significant factors? Check significance
significant_factors = 0
if p_hours < alpha:
    significant_factors += 1
if p_val_bait < alpha:
    significant_factors += 1
if p_val_camp < alpha:
    significant_factors += 1

# Add points for significant factors
evidence_score += significant_factors * 15

# Model quality (R-squared)
evidence_score += min(30, int(r2 * 100))

# Cap at 100
evidence_score = min(100, evidence_score)

explanation = (
    f"Analysis shows visitors catch an average of {approach1_mean:.2f} fish per hour (median: {approach1_median:.2f}). "
    f"The number of hours spent fishing shows a {'significant' if p_hours < alpha else 'weak'} positive correlation "
    f"(r={corr_hours:.2f}, p={p_hours:.4f}). "
    f"Key factors influencing catch: "
    f"using livebait {'significantly increases' if p_val_bait < alpha else 'does not significantly affect'} fish caught (p={p_val_bait:.4f}), "
    f"having a camper {'significantly affects' if p_val_camp < alpha else 'has no significant effect on'} catch (p={p_val_camp:.4f}). "
    f"Linear regression model (R²={r2:.3f}) estimates {hours_coefficient:.3f} additional fish per hour when controlling for other factors. "
    f"The data supports estimating catch rates and identifying influential factors."
)

print(f"\nResponse Score: {evidence_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": evidence_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
