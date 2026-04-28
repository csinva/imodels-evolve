import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Calculate nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Replace infinite values (when seconds = 0) with NaN
df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN efficiency
df_clean = df.dropna(subset=['efficiency'])

print("="*60)
print("DATA EXPLORATION")
print("="*60)
print(f"\nDataset shape: {df_clean.shape}")
print(f"\nBasic statistics:")
print(df_clean[['age', 'nuts_opened', 'seconds', 'efficiency']].describe())

print(f"\nSex distribution:")
print(df_clean['sex'].value_counts())

print(f"\nHelp distribution:")
print(df_clean['help'].value_counts())

# Encode categorical variables
df_clean['sex_encoded'] = df_clean['sex'].map({'m': 1, 'f': 0})
df_clean['help_encoded'] = df_clean['help'].map({'y': 1, 'N': 0})

# Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
correlation_vars = ['age', 'sex_encoded', 'help_encoded', 'efficiency']
corr_matrix = df_clean[correlation_vars].corr()
print("\nCorrelation with efficiency:")
print(corr_matrix['efficiency'].sort_values(ascending=False))

# Statistical tests for each variable
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

# 1. Age influence
print("\n1. AGE INFLUENCE:")
age_corr, age_pval = stats.pearsonr(df_clean['age'], df_clean['efficiency'])
print(f"   Pearson correlation: r={age_corr:.4f}, p-value={age_pval:.4f}")
if age_pval < 0.05:
    print(f"   → Significant relationship (p < 0.05)")
else:
    print(f"   → No significant relationship (p >= 0.05)")

# 2. Sex influence
print("\n2. SEX INFLUENCE:")
males = df_clean[df_clean['sex'] == 'm']['efficiency']
females = df_clean[df_clean['sex'] == 'f']['efficiency']
sex_ttest, sex_pval = stats.ttest_ind(males, females)
print(f"   Male efficiency: mean={males.mean():.4f}, std={males.std():.4f}, n={len(males)}")
print(f"   Female efficiency: mean={females.mean():.4f}, std={females.std():.4f}, n={len(females)}")
print(f"   T-test: t={sex_ttest:.4f}, p-value={sex_pval:.4f}")
if sex_pval < 0.05:
    print(f"   → Significant difference (p < 0.05)")
else:
    print(f"   → No significant difference (p >= 0.05)")

# 3. Help influence
print("\n3. HELP INFLUENCE:")
with_help = df_clean[df_clean['help'] == 'y']['efficiency']
without_help = df_clean[df_clean['help'] == 'N']['efficiency']
help_ttest, help_pval = stats.ttest_ind(with_help, without_help)
print(f"   With help: mean={with_help.mean():.4f}, std={with_help.std():.4f}, n={len(with_help)}")
print(f"   Without help: mean={without_help.mean():.4f}, std={without_help.std():.4f}, n={len(without_help)}")
print(f"   T-test: t={help_ttest:.4f}, p-value={help_pval:.4f}")
if help_pval < 0.05:
    print(f"   → Significant difference (p < 0.05)")
else:
    print(f"   → No significant difference (p >= 0.05)")

# Multiple regression with statsmodels for p-values
print("\n" + "="*60)
print("MULTIPLE REGRESSION ANALYSIS")
print("="*60)

X = df_clean[['age', 'sex_encoded', 'help_encoded']]
y = df_clean['efficiency']

# Add constant for statsmodels
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print("\nOLS Regression Results:")
print(model.summary())

# Interpretable model using ExplainableBoostingRegressor
print("\n" + "="*60)
print("EXPLAINABLE BOOSTING REGRESSOR")
print("="*60)

ebr = ExplainableBoostingRegressor(random_state=42)
ebr.fit(X, y)

# Get feature importances
feature_names = ['age', 'sex', 'help']
importances = ebr.term_importances()
print("\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"   {name}: {imp:.4f}")

# Decision logic based on statistical tests
print("\n" + "="*60)
print("FINAL CONCLUSION")
print("="*60)

# Count significant factors
significant_factors = []
if age_pval < 0.05:
    significant_factors.append('age')
if sex_pval < 0.05:
    significant_factors.append('sex')
if help_pval < 0.05:
    significant_factors.append('help')

print(f"\nSignificant factors (p < 0.05): {significant_factors if significant_factors else 'None'}")

# Generate response score and explanation
if len(significant_factors) == 3:
    # All three factors are significant
    response = 95
    explanation = f"All three factors significantly influence nut-cracking efficiency. Age (p={age_pval:.4f}), sex (p={sex_pval:.4f}), and help (p={help_pval:.4f}) all show statistical significance (p<0.05). The multiple regression model confirms these effects."
elif len(significant_factors) == 2:
    # Two factors are significant
    response = 75
    explanation = f"Two of the three factors significantly influence nut-cracking efficiency: {', '.join(significant_factors)}. "
    if 'age' not in significant_factors:
        explanation += f"Age shows no significant effect (p={age_pval:.4f}). "
    if 'sex' not in significant_factors:
        explanation += f"Sex shows no significant effect (p={sex_pval:.4f}). "
    if 'help' not in significant_factors:
        explanation += f"Help shows no significant effect (p={help_pval:.4f}). "
elif len(significant_factors) == 1:
    # Only one factor is significant
    response = 45
    explanation = f"Only {significant_factors[0]} shows a significant influence (p<0.05) on nut-cracking efficiency. The other factors (age p={age_pval:.4f}, sex p={sex_pval:.4f}, help p={help_pval:.4f}) do not show statistical significance."
else:
    # No significant factors
    response = 10
    explanation = f"None of the three factors show statistically significant influence on nut-cracking efficiency (age p={age_pval:.4f}, sex p={sex_pval:.4f}, help p={help_pval:.4f}). All p-values exceed the 0.05 threshold."

print(f"\nResponse Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*60)
print("Analysis complete! conclusion.txt has been created.")
print("="*60)
