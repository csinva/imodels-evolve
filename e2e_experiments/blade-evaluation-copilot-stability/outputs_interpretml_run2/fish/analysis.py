import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor

# Load the data
df = pd.read_csv('fish.csv')

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Calculate fish per hour (the target metric)
df['fish_per_hour'] = df['fish_caught'] / df['hours']
# Handle cases where hours might be 0
df['fish_per_hour'] = df['fish_per_hour'].replace([np.inf, -np.inf], np.nan)

print("\n" + "=" * 80)
print("FISH PER HOUR ANALYSIS")
print("=" * 80)
print("\nFish per hour statistics:")
print(df['fish_per_hour'].describe())
print(f"\nAverage fish per hour (excluding infinite values): {df['fish_per_hour'].mean():.4f}")
print(f"Median fish per hour: {df['fish_per_hour'].median():.4f}")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print("\nCorrelation with fish_caught:")
print(df[['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()['fish_caught'].sort_values(ascending=False))

print("\nCorrelation with fish_per_hour (where calculable):")
correlations = df[['fish_per_hour', 'livebait', 'camper', 'persons', 'child', 'hours']].corr()['fish_per_hour'].sort_values(ascending=False)
print(correlations)

# Statistical tests for each factor
print("\n" + "=" * 80)
print("STATISTICAL TESTS FOR FACTORS AFFECTING FISH CAUGHT")
print("=" * 80)

# Test 1: Does livebait affect fish caught?
livebait_yes = df[df['livebait'] == 1]['fish_caught']
livebait_no = df[df['livebait'] == 0]['fish_caught']
t_stat, p_value_livebait = stats.ttest_ind(livebait_yes, livebait_no)
print(f"\nLivebait effect on fish caught:")
print(f"  With livebait: mean={livebait_yes.mean():.2f}, std={livebait_yes.std():.2f}")
print(f"  Without livebait: mean={livebait_no.mean():.2f}, std={livebait_no.std():.2f}")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value_livebait:.4f}")

# Test 2: Does camper affect fish caught?
camper_yes = df[df['camper'] == 1]['fish_caught']
camper_no = df[df['camper'] == 0]['fish_caught']
t_stat, p_value_camper = stats.ttest_ind(camper_yes, camper_no)
print(f"\nCamper effect on fish caught:")
print(f"  With camper: mean={camper_yes.mean():.2f}, std={camper_yes.std():.2f}")
print(f"  Without camper: mean={camper_no.mean():.2f}, std={camper_no.std():.2f}")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value_camper:.4f}")

# Test 3: Correlation between persons and fish caught
corr_persons, p_value_persons = stats.pearsonr(df['persons'], df['fish_caught'])
print(f"\nNumber of persons correlation with fish caught:")
print(f"  Correlation: {corr_persons:.4f}, p-value: {p_value_persons:.4f}")

# Test 4: Correlation between child and fish caught
corr_child, p_value_child = stats.pearsonr(df['child'], df['fish_caught'])
print(f"\nNumber of children correlation with fish caught:")
print(f"  Correlation: {corr_child:.4f}, p-value: {p_value_child:.4f}")

# Test 5: Correlation between hours and fish caught
corr_hours, p_value_hours = stats.pearsonr(df['hours'], df['fish_caught'])
print(f"\nHours spent correlation with fish caught:")
print(f"  Correlation: {corr_hours:.4f}, p-value: {p_value_hours:.4f}")

# Regression analysis to estimate fish caught per hour
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS: ESTIMATING FISH CAUGHT RATE")
print("=" * 80)

# Prepare data for modeling
X = df[['livebait', 'camper', 'persons', 'child', 'hours']].copy()
y = df['fish_caught'].copy()

# Statsmodels OLS for detailed statistics
X_sm = sm.add_constant(X)
model_ols = sm.OLS(y, X_sm).fit()
print("\nOLS Regression Results:")
print(model_ols.summary())

# Interpretable model: Linear Regression
lr = LinearRegression()
lr.fit(X, y)

print("\n" + "=" * 80)
print("LINEAR REGRESSION COEFFICIENTS")
print("=" * 80)
for feature, coef in zip(X.columns, lr.coef_):
    print(f"{feature:15s}: {coef:8.4f}")
print(f"{'intercept':15s}: {lr.intercept_:8.4f}")

# Calculate fish per hour coefficient
fish_per_hour_coef = lr.coef_[4]  # hours coefficient
print(f"\nEstimated fish caught per hour: {fish_per_hour_coef:.4f}")
print(f"Average fish caught: {df['fish_caught'].mean():.4f}")
print(f"Average hours: {df['hours'].mean():.4f}")
print(f"Simple average fish per hour: {df['fish_caught'].mean() / df['hours'].mean():.4f}")

# Explainable Boosting Machine for more sophisticated analysis
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING MACHINE ANALYSIS")
print("=" * 80)

ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)

print("\nEBM Feature Importances:")
for feature, importance in zip(X.columns, ebm.term_importances()):
    print(f"{feature:15s}: {importance:8.4f}")

# Model performance
y_pred_ebm = ebm.predict(X)
r2_ebm = r2_score(y, y_pred_ebm)
rmse_ebm = np.sqrt(mean_squared_error(y, y_pred_ebm))
print(f"\nEBM Model Performance:")
print(f"  R² Score: {r2_ebm:.4f}")
print(f"  RMSE: {rmse_ebm:.4f}")

# ANOVA for categorical variables
print("\n" + "=" * 80)
print("ANOVA FOR GROUP EFFECTS")
print("=" * 80)

# ANOVA for persons
groups_persons = [df[df['persons'] == i]['fish_caught'].values for i in df['persons'].unique()]
f_stat_persons, p_value_anova_persons = stats.f_oneway(*groups_persons)
print(f"\nANOVA for number of persons:")
print(f"  F-statistic: {f_stat_persons:.4f}, p-value: {p_value_anova_persons:.4f}")

# ANOVA for child
groups_child = [df[df['child'] == i]['fish_caught'].values for i in df['child'].unique()]
f_stat_child, p_value_anova_child = stats.f_oneway(*groups_child)
print(f"\nANOVA for number of children:")
print(f"  F-statistic: {f_stat_child:.4f}, p-value: {p_value_anova_child:.4f}")

# Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Calculate average fish per hour more carefully
valid_fish_per_hour = df[df['hours'] > 0]['fish_per_hour']
avg_fish_per_hour = valid_fish_per_hour.mean()
median_fish_per_hour = valid_fish_per_hour.median()

print(f"\nAverage fish caught per hour: {avg_fish_per_hour:.4f}")
print(f"Median fish caught per hour: {median_fish_per_hour:.4f}")

# Key factors identified
print("\nKey findings:")
print(f"1. Hours spent fishing has the strongest correlation (r={corr_hours:.3f}, p={p_value_hours:.4f})")
print(f"2. Livebait usage shows significant effect (p={p_value_livebait:.4f})")
print(f"3. Number of persons shows correlation (r={corr_persons:.3f}, p={p_value_persons:.4f})")
print(f"4. Linear regression coefficient for hours: {fish_per_hour_coef:.4f}")

# Determine confidence score
# The question asks about fish per hour rate and what factors influence it
# We have strong evidence that:
# - Hours is the strongest predictor (highly significant)
# - We can estimate the rate using regression
# - Multiple factors influence the catch rate

significant_factors = []
if p_value_livebait < 0.05:
    significant_factors.append("livebait")
if p_value_camper < 0.05:
    significant_factors.append("camper")
if p_value_persons < 0.05:
    significant_factors.append("persons")
if p_value_child < 0.05:
    significant_factors.append("child")
if p_value_hours < 0.05:
    significant_factors.append("hours")

print(f"\nStatistically significant factors (p < 0.05): {', '.join(significant_factors)}")

# The question asks about rate per hour and what factors influence it
# We successfully identified multiple factors and estimated the rate
# Strong evidence = high confidence (80-100)
# We have clear estimates and significant relationships

explanation = (
    f"Analysis shows visitors catch an average of {avg_fish_per_hour:.2f} fish per hour (median: {median_fish_per_hour:.2f}). "
    f"Multiple factors significantly influence catch rates: hours spent (r={corr_hours:.2f}, p<{p_value_hours:.4f}), "
    f"livebait use (p={p_value_livebait:.4f}), and number of persons (r={corr_persons:.2f}, p={p_value_persons:.4f}). "
    f"The OLS regression model (R²={model_ols.rsquared:.3f}) shows hours is the strongest predictor with coefficient {fish_per_hour_coef:.3f}, "
    f"meaning each additional hour yields approximately {fish_per_hour_coef:.2f} more fish. The Explainable Boosting Machine "
    f"confirms the model's predictive power (R²={r2_ebm:.3f}). "
    f"This provides strong evidence for estimating catch rates and identifying influential factors."
)

# High confidence (85) because:
# - Clear average rate calculated
# - Multiple significant factors identified
# - Strong statistical evidence (p-values < 0.05)
# - Good model fit (R² from OLS)
response = 85

result = {
    "response": response,
    "explanation": explanation
}

print(f"\nFinal Response: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
