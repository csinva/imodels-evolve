import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Calculate nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']
df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0)  # Handle division by zero

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("EFFICIENCY BY CATEGORICAL VARIABLES")
print("=" * 80)
print(f"\nEfficiency by Sex:")
print(df.groupby('sex')['efficiency'].agg(['mean', 'std', 'count']))

print(f"\nEfficiency by Help:")
print(df.groupby('help')['efficiency'].agg(['mean', 'std', 'count']))

print(f"\nEfficiency by Age:")
print(df.groupby('age')['efficiency'].agg(['mean', 'std', 'count']))

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print(f"\nCorrelation between age and efficiency:")
age_efficiency_corr = stats.pearsonr(df['age'], df['efficiency'])
print(f"Pearson r = {age_efficiency_corr[0]:.4f}, p-value = {age_efficiency_corr[1]:.4f}")

print(f"\nCorrelation between age and nuts_opened:")
age_nuts_corr = stats.pearsonr(df['age'], df['nuts_opened'])
print(f"Pearson r = {age_nuts_corr[0]:.4f}, p-value = {age_nuts_corr[1]:.4f}")

# T-test for sex
print("\n" + "=" * 80)
print("T-TEST: SEX AND EFFICIENCY")
print("=" * 80)
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
sex_ttest = stats.ttest_ind(male_eff, female_eff)
print(f"Male efficiency: mean={male_eff.mean():.4f}, std={male_eff.std():.4f}, n={len(male_eff)}")
print(f"Female efficiency: mean={female_eff.mean():.4f}, std={female_eff.std():.4f}, n={len(female_eff)}")
print(f"T-statistic = {sex_ttest[0]:.4f}, p-value = {sex_ttest[1]:.4f}")

# T-test for help
print("\n" + "=" * 80)
print("T-TEST: HELP AND EFFICIENCY")
print("=" * 80)
help_yes = df[df['help'] == 'y']['efficiency']
help_no = df[df['help'] == 'N']['efficiency']
help_ttest = stats.ttest_ind(help_yes, help_no)
print(f"With help: mean={help_yes.mean():.4f}, std={help_yes.std():.4f}, n={len(help_yes)}")
print(f"Without help: mean={help_no.mean():.4f}, std={help_no.std():.4f}, n={len(help_no)}")
print(f"T-statistic = {help_ttest[0]:.4f}, p-value = {help_ttest[1]:.4f}")

# Prepare data for regression
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION ANALYSIS")
print("=" * 80)

# Encode categorical variables
le_sex = LabelEncoder()
le_help = LabelEncoder()
df['sex_encoded'] = le_sex.fit_transform(df['sex'])
df['help_encoded'] = le_help.fit_transform(df['help'])

# Build regression model with statsmodels for p-values
X = df[['age', 'sex_encoded', 'help_encoded']]
y = df['efficiency']
X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const).fit()
print(model_ols.summary())

# Extract key statistics
print("\n" + "=" * 80)
print("REGRESSION COEFFICIENTS AND SIGNIFICANCE")
print("=" * 80)
for col in ['const', 'age', 'sex_encoded', 'help_encoded']:
    coef = model_ols.params[col]
    pval = model_ols.pvalues[col]
    print(f"{col:15s}: coef={coef:8.5f}, p-value={pval:.4f}, significant={'YES' if pval < 0.05 else 'NO'}")

# Use Explainable Boosting Regressor for feature importance
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING REGRESSOR")
print("=" * 80)
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X, y)
print(f"R-squared: {ebm.score(X, y):.4f}")
print(f"\nFeature importances:")
for i, col in enumerate(['age', 'sex_encoded', 'help_encoded']):
    print(f"  {col:15s}: {ebm.term_importances()[i]:.4f}")

# Analyze results and create conclusion
print("\n" + "=" * 80)
print("INTERPRETATION AND CONCLUSION")
print("=" * 80)

# Check statistical significance
age_significant = model_ols.pvalues['age'] < 0.05
sex_significant = model_ols.pvalues['sex_encoded'] < 0.05
help_significant = model_ols.pvalues['help_encoded'] < 0.05

print(f"\nStatistical Significance Results:")
print(f"  Age: {'SIGNIFICANT' if age_significant else 'NOT SIGNIFICANT'} (p={model_ols.pvalues['age']:.4f})")
print(f"  Sex: {'SIGNIFICANT' if sex_significant else 'NOT SIGNIFICANT'} (p={model_ols.pvalues['sex_encoded']:.4f})")
print(f"  Help: {'SIGNIFICANT' if help_significant else 'NOT SIGNIFICANT'} (p={model_ols.pvalues['help_encoded']:.4f})")

# Count significant predictors
num_significant = sum([age_significant, sex_significant, help_significant])

# Determine response score
if num_significant == 3:
    response_score = 95
    explanation = f"All three factors show statistically significant relationships with nut-cracking efficiency. Age (p={model_ols.pvalues['age']:.4f}), sex (p={model_ols.pvalues['sex_encoded']:.4f}), and receiving help (p={model_ols.pvalues['help_encoded']:.4f}) all significantly influence efficiency. The regression model (R²={model_ols.rsquared:.3f}) confirms that these variables collectively explain the variation in nut-cracking performance."
elif num_significant == 2:
    sig_vars = []
    if age_significant:
        sig_vars.append(f"age (p={model_ols.pvalues['age']:.4f})")
    if sex_significant:
        sig_vars.append(f"sex (p={model_ols.pvalues['sex_encoded']:.4f})")
    if help_significant:
        sig_vars.append(f"help (p={model_ols.pvalues['help_encoded']:.4f})")
    response_score = 70
    explanation = f"Two of the three factors show statistically significant relationships: {' and '.join(sig_vars)}. The remaining factor does not show a significant effect (p>0.05). Overall, there is strong evidence that some of these factors influence nut-cracking efficiency."
elif num_significant == 1:
    if age_significant:
        sig_var = f"age (p={model_ols.pvalues['age']:.4f})"
    elif sex_significant:
        sig_var = f"sex (p={model_ols.pvalues['sex_encoded']:.4f})"
    else:
        sig_var = f"help (p={model_ols.pvalues['help_encoded']:.4f})"
    response_score = 45
    explanation = f"Only one factor, {sig_var}, shows a statistically significant relationship with nut-cracking efficiency. The other two factors do not demonstrate significant effects (p>0.05). Evidence for the combined influence of all three factors is weak."
else:
    response_score = 10
    explanation = f"None of the three factors (age p={model_ols.pvalues['age']:.4f}, sex p={model_ols.pvalues['sex_encoded']:.4f}, help p={model_ols.pvalues['help_encoded']:.4f}) show statistically significant relationships with nut-cracking efficiency at the conventional α=0.05 level. There is insufficient evidence to support that these factors influence nut-cracking efficiency."

print(f"\nFinal Assessment:")
print(f"  Response Score: {response_score}/100")
print(f"  Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
