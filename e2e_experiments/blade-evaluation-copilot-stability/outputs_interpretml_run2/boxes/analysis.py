import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingClassifier

# Load data
df = pd.read_csv('boxes.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nDataset summary:")
print(df.describe())

# Research question: How do children's reliance on majority preference develop 
# over growth in age across different cultural contexts?

# The outcome variable y has 3 values:
# 1 = unchosen option
# 2 = majority option (this indicates reliance on majority preference)
# 3 = minority option

# Create binary indicator for following majority
df['follows_majority'] = (df['y'] == 2).astype(int)

print("\n" + "="*80)
print("EXPLORATORY ANALYSIS")
print("="*80)

# Check proportion following majority by age
print("\nProportion following majority by age:")
majority_by_age = df.groupby('age')['follows_majority'].agg(['mean', 'count'])
print(majority_by_age)

# Check proportion following majority by culture
print("\nProportion following majority by culture:")
majority_by_culture = df.groupby('culture')['follows_majority'].agg(['mean', 'count'])
print(majority_by_culture)

# Check correlation between age and following majority
corr, p_value = stats.spearmanr(df['age'], df['follows_majority'])
print(f"\nSpearman correlation between age and following majority: r={corr:.4f}, p={p_value:.6f}")

print("\n" + "="*80)
print("STATISTICAL MODELING")
print("="*80)

# Model 1: Logistic regression with age
print("\nModel 1: Age only")
X_age = df[['age']].values
y = df['follows_majority'].values
scaler = StandardScaler()
X_age_scaled = scaler.fit_transform(X_age)
model_age = LogisticRegression(random_state=42)
model_age.fit(X_age_scaled, y)
print(f"Age coefficient: {model_age.coef_[0][0]:.4f}")

# Use statsmodels for p-values
logit_model_age = smf.logit('follows_majority ~ age', data=df).fit(disp=0)
print(logit_model_age.summary2().tables[1])

# Model 2: Age + Culture interaction
print("\nModel 2: Age * Culture interaction")
logit_model_interaction = smf.logit('follows_majority ~ age * C(culture)', data=df).fit(disp=0)
print(logit_model_interaction.summary2().tables[1])

# Model 3: Explainable Boosting Classifier for better interpretability
print("\nModel 3: Explainable Boosting Classifier")
X_ebm = df[['age', 'culture', 'gender', 'majority_first']].values
ebm = ExplainableBoostingClassifier(random_state=42, max_rounds=100)
ebm.fit(X_ebm, y)

# Get feature importances
importances = ebm.term_importances()
feature_names = ['age', 'culture', 'gender', 'majority_first']
print("\nFeature importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

print("\n" + "="*80)
print("STATISTICAL TESTS FOR AGE-CULTURE INTERACTION")
print("="*80)

# Test for interaction: Does age effect differ by culture?
# Compare models with and without interaction
logit_model_main = smf.logit('follows_majority ~ age + C(culture)', data=df).fit(disp=0)
logit_model_interaction = smf.logit('follows_majority ~ age * C(culture)', data=df).fit(disp=0)

# Likelihood ratio test
lr_stat = -2 * (logit_model_main.llf - logit_model_interaction.llf)
df_diff = logit_model_interaction.df_model - logit_model_main.df_model
p_value_lr = stats.chi2.sf(lr_stat, df_diff)

print(f"\nLikelihood ratio test for age*culture interaction:")
print(f"  LR statistic: {lr_stat:.4f}")
print(f"  df: {df_diff}")
print(f"  p-value: {p_value_lr:.6f}")

# Test age effect within each culture
print("\nAge effect within each culture:")
culture_age_results = []
for culture_id in sorted(df['culture'].unique()):
    df_culture = df[df['culture'] == culture_id]
    if len(df_culture) > 10:  # Need enough data
        corr, p = stats.spearmanr(df_culture['age'], df_culture['follows_majority'])
        culture_age_results.append({
            'culture': culture_id,
            'n': len(df_culture),
            'corr': corr,
            'p_value': p
        })
        print(f"  Culture {culture_id}: n={len(df_culture)}, r={corr:.4f}, p={p:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine response based on statistical evidence
# Key question: Do children's reliance on majority preference develop 
# over age AND does this differ across cultural contexts?

# Evidence for age effect
age_effect_significant = logit_model_age.pvalues['age'] < 0.05
age_coef = logit_model_age.params['age']

# Evidence for interaction
interaction_significant = p_value_lr < 0.05

# Check if age effects vary across cultures
culture_variations = [r for r in culture_age_results if r['p_value'] < 0.05]
significant_cultures = len(culture_variations)
total_cultures = len(culture_age_results)

print(f"\nAge main effect: coefficient={age_coef:.4f}, p={logit_model_age.pvalues['age']:.4f}, significant={age_effect_significant}")
print(f"Age*Culture interaction: p={p_value_lr:.4f}, significant={interaction_significant}")
print(f"Cultures with significant age effect: {significant_cultures}/{total_cultures}")

# Formulate response
if interaction_significant:
    # Strong evidence for developmental differences across cultures
    response = 85
    explanation = (
        f"Strong evidence that children's reliance on majority preference develops with age "
        f"differently across cultural contexts. The age*culture interaction is statistically "
        f"significant (p={p_value_lr:.4f}). Age effect significant in {significant_cultures} of "
        f"{total_cultures} cultures tested individually. Main age effect: coefficient={age_coef:.3f}, "
        f"p={logit_model_age.pvalues['age']:.4f}."
    )
elif age_effect_significant and significant_cultures >= 2:
    # Moderate evidence: age matters but interaction test not significant
    response = 65
    explanation = (
        f"Moderate evidence for developmental differences across cultures. Age has a significant "
        f"main effect (coef={age_coef:.3f}, p={logit_model_age.pvalues['age']:.4f}) and shows "
        f"significant effects in {significant_cultures} cultures individually, though the formal "
        f"interaction test is not significant (p={p_value_lr:.4f})."
    )
elif age_effect_significant:
    # Age matters but weak evidence for cultural differences
    response = 45
    explanation = (
        f"Age has a significant effect on majority preference (coef={age_coef:.3f}, "
        f"p={logit_model_age.pvalues['age']:.4f}), but limited evidence that this develops "
        f"differently across cultures. Age*culture interaction not significant (p={p_value_lr:.4f}), "
        f"and only {significant_cultures} cultures show significant age effects individually."
    )
else:
    # No strong evidence
    response = 25
    explanation = (
        f"Limited evidence for developmental changes in majority preference across cultures. "
        f"Age main effect not significant (p={logit_model_age.pvalues['age']:.4f}), and "
        f"age*culture interaction not significant (p={p_value_lr:.4f})."
    )

print(f"\nResponse: {response}")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
