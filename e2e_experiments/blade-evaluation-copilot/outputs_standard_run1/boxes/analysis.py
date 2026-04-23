import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols, mnlogit
from imodels import RuleFitClassifier, FIGSClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('boxes.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Understand the outcome variable
print("\n" + "=" * 80)
print("OUTCOME VARIABLE DISTRIBUTION")
print("=" * 80)
print("\nOutcome (y) value counts:")
print(df['y'].value_counts().sort_index())
print("\nOutcome meanings: 1=unchosen, 2=majority, 3=minority")

# Key research question: How does reliance on majority preference develop with age?
# We need to look at whether choosing the majority option (y=2) increases with age

# Create binary indicators
df['chose_majority'] = (df['y'] == 2).astype(int)
df['chose_minority'] = (df['y'] == 3).astype(int)
df['chose_unchosen'] = (df['y'] == 1).astype(int)

print("\n" + "=" * 80)
print("MAJORITY PREFERENCE BY AGE")
print("=" * 80)
print("\nProportion choosing majority by age:")
majority_by_age = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print(majority_by_age)

# Statistical test: Correlation between age and majority choice
corr, p_value = stats.spearmanr(df['age'], df['chose_majority'])
print(f"\nSpearman correlation between age and choosing majority: r={corr:.4f}, p={p_value:.4f}")

# Logistic regression: Age predicting majority choice
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION: AGE PREDICTING MAJORITY CHOICE")
print("=" * 80)
X = sm.add_constant(df['age'])
logit_model = sm.Logit(df['chose_majority'], X)
logit_result = logit_model.fit(disp=0)
print(logit_result.summary())

# Extract key statistics
age_coef = logit_result.params['age']
age_pval = logit_result.pvalues['age']
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pval:.4f}")

# Check interaction with culture
print("\n" + "=" * 80)
print("CULTURAL CONTEXT ANALYSIS")
print("=" * 80)
print("\nProportion choosing majority by culture:")
majority_by_culture = df.groupby('culture')['chose_majority'].agg(['mean', 'count'])
print(majority_by_culture)

# ANOVA to test if culture matters
culture_groups = [df[df['culture'] == c]['chose_majority'] for c in df['culture'].unique()]
f_stat, p_value_culture = stats.f_oneway(*culture_groups)
print(f"\nANOVA for culture effect on majority choice: F={f_stat:.4f}, p={p_value_culture:.4f}")

# Interaction model: Age × Culture
print("\n" + "=" * 80)
print("INTERACTION MODEL: AGE × CULTURE")
print("=" * 80)

# Create interaction term
df['age_x_culture'] = df['age'] * df['culture']

# Logistic regression with interaction
X_interact = sm.add_constant(df[['age', 'culture', 'age_x_culture']])
logit_interact = sm.Logit(df['chose_majority'], X_interact)
interact_result = logit_interact.fit(disp=0)
print(interact_result.summary())

interaction_coef = interact_result.params['age_x_culture']
interaction_pval = interact_result.pvalues['age_x_culture']
print(f"\nAge × Culture interaction coefficient: {interaction_coef:.4f}, p-value: {interaction_pval:.4f}")

# Age effect by culture
print("\n" + "=" * 80)
print("AGE EFFECT WITHIN EACH CULTURE")
print("=" * 80)
age_effects = []
for culture in sorted(df['culture'].unique()):
    culture_data = df[df['culture'] == culture]
    if len(culture_data) > 10:  # Only analyze cultures with sufficient data
        corr_c, p_c = stats.spearmanr(culture_data['age'], culture_data['chose_majority'])
        age_effects.append({
            'culture': culture,
            'n': len(culture_data),
            'correlation': corr_c,
            'p_value': p_c
        })
        print(f"Culture {culture}: n={len(culture_data)}, correlation={corr_c:.4f}, p={p_c:.4f}")

# Interpretable model using imodels
print("\n" + "=" * 80)
print("INTERPRETABLE MODEL (FIGS Classifier)")
print("=" * 80)

# Prepare features
X_features = df[['age', 'culture', 'gender', 'majority_first']].values
y_target = df['chose_majority'].values

try:
    figs = FIGSClassifier(max_rules=5)
    figs.fit(X_features, y_target)
    print("\nFIGS Rules:")
    print(figs)
    
    # Feature importance
    if hasattr(figs, 'feature_importances_'):
        feature_names = ['age', 'culture', 'gender', 'majority_first']
        importances = figs.feature_importances_
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.4f}")
except Exception as e:
    print(f"FIGS model failed: {e}")

# Overall interpretation
print("\n" + "=" * 80)
print("OVERALL INTERPRETATION")
print("=" * 80)

# Main effect of age
if age_pval < 0.05:
    direction = "positive" if age_coef > 0 else "negative"
    print(f"✓ Age has a significant {direction} effect on majority choice (p={age_pval:.4f})")
else:
    print(f"✗ Age does NOT have a significant effect on majority choice (p={age_pval:.4f})")

# Interaction with culture
if interaction_pval < 0.05:
    print(f"✓ The relationship between age and majority preference varies significantly across cultures (p={interaction_pval:.4f})")
    different_cultures = True
else:
    print(f"✗ The relationship between age and majority preference does NOT vary significantly across cultures (p={interaction_pval:.4f})")
    different_cultures = False

# Check consistency across cultures
sig_cultures = sum(1 for effect in age_effects if effect['p_value'] < 0.05)
total_cultures = len(age_effects)
print(f"\nAge effect is significant in {sig_cultures}/{total_cultures} cultures with sufficient data")

# Determine final answer
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Research question: "How do children's reliance on majority preference develop over growth in age across different cultural contexts?"
# This is asking if there's a RELATIONSHIP that varies by culture

# Evidence for development over age
age_develops = age_pval < 0.05

# Evidence for cultural variation in this development
cultural_variation = interaction_pval < 0.05 or (p_value_culture < 0.05 and sig_cultures < total_cultures * 0.8)

if age_develops and cultural_variation:
    # Strong evidence for both age effect and cultural differences
    response = 85
    explanation = f"Strong evidence that majority preference develops with age (p={age_pval:.4f}, coef={age_coef:.4f}) AND varies across cultures. Age×Culture interaction: p={interaction_pval:.4f}. Age effect significant in {sig_cultures}/{total_cultures} cultures, suggesting cultural context moderates development."
elif age_develops and not cultural_variation:
    # Age effect but similar across cultures
    response = 60
    explanation = f"Majority preference develops with age (p={age_pval:.4f}, coef={age_coef:.4f}), but development is similar across cultures (interaction p={interaction_pval:.4f}). Age effect consistent across {sig_cultures}/{total_cultures} cultures tested."
elif not age_develops and cultural_variation:
    # No overall age effect but cultural differences exist
    response = 40
    explanation = f"No significant overall age effect (p={age_pval:.4f}), though cultures differ in majority preference (p={p_value_culture:.4f}). Mixed age effects across cultures suggest complex cultural moderation that doesn't produce clear developmental trend."
else:
    # Neither age effect nor cultural variation
    response = 15
    explanation = f"No significant age effect (p={age_pval:.4f}) and no significant cultural variation in development (interaction p={interaction_pval:.4f}). Limited evidence for developmental changes in majority preference across cultural contexts."

print(f"\nResponse: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("conclusion.txt written successfully!")
print("=" * 80)
