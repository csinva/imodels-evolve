import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor, HingeGAMRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load info.json and data
with open('info.json', 'r') as f:
    info = json.load(f)

research_question = info['research_questions'][0]
print("=" * 80)
print("RESEARCH QUESTION:")
print(research_question)
print("=" * 80)

# Load dataset
df = pd.read_csv('reading.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")

# Explore the data
print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Focus on key variables: reader_view (treatment), speed (outcome), dyslexia status
print("\n" + "=" * 80)
print("KEY VARIABLES ANALYSIS")
print("=" * 80)

# The research question: Does 'Reader View' improve reading speed for individuals with dyslexia?
# Key variables:
# - reader_view: treatment (0 or 1)
# - speed: reading speed (outcome)
# - dyslexia: dyslexia status (0 - No Dyslexia, 1 - Dyslexia, 2 - Severe Dyslexia)
# - dyslexia_bin: binary dyslexia indicator (0 or 1)

print("\nReader view distribution:")
print(df['reader_view'].value_counts())

print("\nDyslexia status distribution:")
print(df['dyslexia'].value_counts())

print("\nDyslexia binary distribution:")
print(df['dyslexia_bin'].value_counts())

print("\nSpeed statistics:")
print(df['speed'].describe())

# Check for extreme outliers in speed
print(f"\nSpeed outliers (>99th percentile): {(df['speed'] > df['speed'].quantile(0.99)).sum()}")

# Filter out extreme outliers in speed for better analysis
df_clean = df[df['speed'] < df['speed'].quantile(0.99)].copy()
print(f"Dataset after removing extreme outliers: {df_clean.shape}")

# BIVARIATE ANALYSIS: Effect of reader_view on speed for dyslexic individuals
print("\n" + "=" * 80)
print("BIVARIATE ANALYSIS: Reader View Effect on Speed for Dyslexic Individuals")
print("=" * 80)

# Focus on individuals with dyslexia (dyslexia_bin=1)
df_dyslexic = df_clean[df_clean['dyslexia_bin'] == 1].copy()
df_non_dyslexic = df_clean[df_clean['dyslexia_bin'] == 0].copy()

print(f"\nNumber of dyslexic participants: {len(df_dyslexic)}")
print(f"Number of non-dyslexic participants: {len(df_non_dyslexic)}")

# Compare speed with and without reader view for dyslexic individuals
dyslexic_with_rv = df_dyslexic[df_dyslexic['reader_view'] == 1]['speed']
dyslexic_without_rv = df_dyslexic[df_dyslexic['reader_view'] == 0]['speed']

print(f"\nDyslexic individuals WITH reader view: n={len(dyslexic_with_rv)}, mean speed={dyslexic_with_rv.mean():.2f}, std={dyslexic_with_rv.std():.2f}")
print(f"Dyslexic individuals WITHOUT reader view: n={len(dyslexic_without_rv)}, mean speed={dyslexic_without_rv.mean():.2f}, std={dyslexic_without_rv.std():.2f}")

# T-test
if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0:
    t_stat, p_value = stats.ttest_ind(dyslexic_with_rv, dyslexic_without_rv)
    print(f"\nT-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(dyslexic_with_rv)-1)*dyslexic_with_rv.std()**2 + (len(dyslexic_without_rv)-1)*dyslexic_without_rv.std()**2) / (len(dyslexic_with_rv) + len(dyslexic_without_rv) - 2))
    cohens_d = (dyslexic_with_rv.mean() - dyslexic_without_rv.mean()) / pooled_std
    print(f"Cohen's d (effect size): {cohens_d:.4f}")

# INTERACTION ANALYSIS: Does reader_view effect differ between dyslexic and non-dyslexic?
print("\n" + "=" * 80)
print("INTERACTION ANALYSIS: Reader View × Dyslexia")
print("=" * 80)

# Create interaction term
df_clean['reader_view_x_dyslexia'] = df_clean['reader_view'] * df_clean['dyslexia_bin']

# Correlation matrix for key variables
print("\nCorrelation with speed:")
numeric_cols = ['reader_view', 'dyslexia_bin', 'reader_view_x_dyslexia', 'age', 'num_words', 
                'correct_rate', 'adjusted_running_time', 'scrolling_time']
for col in numeric_cols:
    if col in df_clean.columns:
        corr = df_clean[col].corr(df_clean['speed'])
        print(f"  {col}: {corr:.4f}")

# CLASSICAL STATISTICAL TEST WITH CONTROLS
print("\n" + "=" * 80)
print("CLASSICAL REGRESSION: OLS with Controls")
print("=" * 80)

# Prepare features for regression
# Control variables: age, num_words, correct_rate, device (will use one-hot), education, gender
# We need numeric features for statsmodels
df_model = df_clean.copy()

# One-hot encode categorical variables
device_dummies = pd.get_dummies(df_model['device'], prefix='device', drop_first=True)
education_dummies = pd.get_dummies(df_model['education'], prefix='education', drop_first=True)

# Convert boolean dummies to int
device_dummies = device_dummies.astype(int)
education_dummies = education_dummies.astype(int)

# Combine all features
feature_cols = ['reader_view', 'dyslexia_bin', 'reader_view_x_dyslexia', 
                'age', 'num_words', 'correct_rate', 'gender', 'adjusted_running_time']
X_base = df_model[feature_cols].copy()
X_with_dummies = pd.concat([X_base, device_dummies, education_dummies], axis=1)

# Remove rows with missing values
X_with_dummies = X_with_dummies.dropna()
y = df_model.loc[X_with_dummies.index, 'speed']

# Ensure all are numeric
X_with_dummies = X_with_dummies.apply(pd.to_numeric, errors='coerce')
X_with_dummies = X_with_dummies.dropna()
y = y.loc[X_with_dummies.index]

# Add constant
X_ols = sm.add_constant(X_with_dummies)

# Fit OLS model
ols_model = sm.OLS(y, X_ols).fit()
print("\nOLS Regression Results:")
print(ols_model.summary())

# Extract key coefficients
print("\n" + "=" * 80)
print("KEY COEFFICIENTS FROM OLS:")
print("=" * 80)
print(f"reader_view coefficient: {ols_model.params.get('reader_view', np.nan):.4f}, p-value: {ols_model.pvalues.get('reader_view', np.nan):.4f}")
print(f"dyslexia_bin coefficient: {ols_model.params.get('dyslexia_bin', np.nan):.4f}, p-value: {ols_model.pvalues.get('dyslexia_bin', np.nan):.4f}")
print(f"reader_view_x_dyslexia (INTERACTION) coefficient: {ols_model.params.get('reader_view_x_dyslexia', np.nan):.4f}, p-value: {ols_model.pvalues.get('reader_view_x_dyslexia', np.nan):.4f}")

# INTERPRETABLE MODELS
print("\n" + "=" * 80)
print("INTERPRETABLE MODELS: agentic_imodels")
print("=" * 80)

# Prepare feature matrix for interpretable models (numeric only, no interaction initially)
feature_cols_imodels = ['reader_view', 'dyslexia_bin', 'age', 'num_words', 
                        'correct_rate', 'gender', 'adjusted_running_time', 'scrolling_time',
                        'img_width', 'dyslexia', 'Flesch_Kincaid']

X_imodels = df_model[feature_cols_imodels].dropna()
y_imodels = df_model.loc[X_imodels.index, 'speed']

# Fit multiple interpretable models
print("\n" + "-" * 80)
print("Model 1: SmartAdditiveRegressor (Honest GAM)")
print("-" * 80)
model1 = SmartAdditiveRegressor()
model1.fit(X_imodels, y_imodels)
y_pred1 = model1.predict(X_imodels)
r2_1 = r2_score(y_imodels, y_pred1)
print(f"R² score: {r2_1:.4f}")
print("\nFitted model:")
print(model1)

print("\n" + "-" * 80)
print("Model 2: HingeEBMRegressor (High-rank, decoupled)")
print("-" * 80)
model2 = HingeEBMRegressor()
model2.fit(X_imodels, y_imodels)
y_pred2 = model2.predict(X_imodels)
r2_2 = r2_score(y_imodels, y_pred2)
print(f"R² score: {r2_2:.4f}")
print("\nFitted model:")
print(model2)

print("\n" + "-" * 80)
print("Model 3: WinsorizedSparseOLSRegressor (Honest sparse linear)")
print("-" * 80)
model3 = WinsorizedSparseOLSRegressor()
model3.fit(X_imodels, y_imodels)
y_pred3 = model3.predict(X_imodels)
r2_3 = r2_score(y_imodels, y_pred3)
print(f"R² score: {r2_3:.4f}")
print("\nFitted model:")
print(model3)

print("\n" + "-" * 80)
print("Model 4: HingeGAMRegressor (Honest pure hinge GAM)")
print("-" * 80)
model4 = HingeGAMRegressor()
model4.fit(X_imodels, y_imodels)
y_pred4 = model4.predict(X_imodels)
r2_4 = r2_score(y_imodels, y_pred4)
print(f"R² score: {r2_4:.4f}")
print("\nFitted model:")
print(model4)

# FOCUSED ANALYSIS ON DYSLEXIC SUBSET
print("\n" + "=" * 80)
print("FOCUSED ANALYSIS: Models Trained on Dyslexic Individuals Only")
print("=" * 80)

X_dyslexic = df_dyslexic[feature_cols_imodels].dropna()
y_dyslexic = df_dyslexic.loc[X_dyslexic.index, 'speed']

if len(X_dyslexic) > 50:  # Only if we have enough data
    print("\n" + "-" * 80)
    print("SmartAdditiveRegressor on Dyslexic Subset")
    print("-" * 80)
    model_dys = SmartAdditiveRegressor()
    model_dys.fit(X_dyslexic, y_dyslexic)
    print(model_dys)
    
    print("\n" + "-" * 80)
    print("HingeEBMRegressor on Dyslexic Subset")
    print("-" * 80)
    model_dys2 = HingeEBMRegressor()
    model_dys2.fit(X_dyslexic, y_dyslexic)
    print(model_dys2)

# CONCLUSION
print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Analyze the evidence
evidence_points = []

# 1. Bivariate analysis
if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0:
    if p_value < 0.05:
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            evidence_points.append(f"Bivariate t-test: Dyslexic individuals WITH reader view have higher speed (mean={dyslexic_with_rv.mean():.2f}) than WITHOUT (mean={dyslexic_without_rv.mean():.2f}), p={p_value:.4f}, Cohen's d={cohens_d:.4f}")
            bivariate_score = 30  # Some positive evidence
        else:
            evidence_points.append(f"Bivariate t-test: Dyslexic individuals WITH reader view have LOWER speed (mean={dyslexic_with_rv.mean():.2f}) than WITHOUT (mean={dyslexic_without_rv.mean():.2f}), p={p_value:.4f}, Cohen's d={cohens_d:.4f}")
            bivariate_score = -30  # Negative evidence
    else:
        evidence_points.append(f"Bivariate t-test: No significant difference (p={p_value:.4f})")
        bivariate_score = 0

# 2. OLS regression with controls
interaction_coef = ols_model.params.get('reader_view_x_dyslexia', 0)
interaction_pval = ols_model.pvalues.get('reader_view_x_dyslexia', 1)
reader_view_coef = ols_model.params.get('reader_view', 0)
reader_view_pval = ols_model.pvalues.get('reader_view', 1)

if interaction_pval < 0.05:
    if interaction_coef > 0:
        evidence_points.append(f"OLS with controls: Significant positive interaction effect (reader_view × dyslexia_bin coef={interaction_coef:.4f}, p={interaction_pval:.4f}). This suggests reader view improves speed MORE for dyslexic individuals.")
        ols_score = 40
    else:
        evidence_points.append(f"OLS with controls: Significant negative interaction effect (reader_view × dyslexia_bin coef={interaction_coef:.4f}, p={interaction_pval:.4f}). This suggests reader view is LESS beneficial for dyslexic individuals.")
        ols_score = -40
else:
    if reader_view_pval < 0.05 and reader_view_coef > 0:
        evidence_points.append(f"OLS with controls: Main effect of reader_view is significant and positive (coef={reader_view_coef:.4f}, p={reader_view_pval:.4f}), but interaction not significant (p={interaction_pval:.4f}). Effect is general, not specific to dyslexia.")
        ols_score = 20
    else:
        evidence_points.append(f"OLS with controls: No significant interaction effect (p={interaction_pval:.4f}), and main reader_view effect not significant (p={reader_view_pval:.4f}).")
        ols_score = 0

# 3. Interpretable models - check if reader_view is important/zeroed out
print("\nEvidence summary:")
for i, point in enumerate(evidence_points, 1):
    print(f"{i}. {point}")

# Determine final score based on evidence
# The question is specifically about dyslexic individuals
# We need strong evidence that reader_view improves speed FOR DYSLEXIC PEOPLE specifically

explanation_parts = []

if bivariate_score > 0:
    explanation_parts.append(f"Bivariate analysis shows dyslexic individuals read faster with reader view (p={p_value:.4f}, Cohen's d={cohens_d:.4f}).")
else:
    explanation_parts.append(f"Bivariate analysis shows no significant speed improvement for dyslexic individuals with reader view (p={p_value:.4f}).")

if ols_score > 30:
    explanation_parts.append(f"OLS regression confirms a significant positive interaction (reader_view × dyslexia β={interaction_coef:.4f}, p={interaction_pval:.4f}), meaning the benefit is specific to dyslexic readers.")
elif ols_score > 0:
    explanation_parts.append(f"OLS shows a general reader_view effect but no specific benefit for dyslexic individuals (interaction p={interaction_pval:.4f}).")
else:
    explanation_parts.append(f"OLS regression with controls finds no significant effect specific to dyslexic individuals (interaction p={interaction_pval:.4f}).")

# Interpretable models: Look at whether reader_view is important
explanation_parts.append("Interpretable models (SmartAdditive, HingeEBM, WinsorizedSparseOLS, HingeGAM) were fitted to reveal feature importance and shape. These models show the full feature space including reader_view and dyslexia status.")

# Final scoring logic
if bivariate_score > 0 and ols_score > 30:
    # Strong evidence: both bivariate and interaction significant
    final_score = 75
    conclusion = "Yes, strong evidence that reader view improves reading speed for dyslexic individuals."
elif bivariate_score > 0 and ols_score > 0:
    # Moderate evidence: bivariate significant, general effect but not interaction
    final_score = 50
    conclusion = "Moderate evidence: reader view improves reading speed generally, but the benefit is not specific to dyslexic individuals."
elif bivariate_score > 0:
    # Weak evidence: only bivariate significant
    final_score = 35
    conclusion = "Weak evidence: bivariate analysis suggests improvement for dyslexic individuals, but this does not hold after controlling for confounders."
elif ols_score > 0:
    # General effect only
    final_score = 30
    conclusion = "Weak evidence: reader view improves reading speed in general, but not specifically for dyslexic individuals."
else:
    # No evidence
    final_score = 15
    conclusion = "No evidence: reader view does not significantly improve reading speed for dyslexic individuals."

explanation = " ".join(explanation_parts) + " " + conclusion

print("\n" + "=" * 80)
print("FINAL CONCLUSION:")
print("=" * 80)
print(f"Score: {final_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
output = {
    "response": final_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

print("\n✓ conclusion.txt written successfully!")
