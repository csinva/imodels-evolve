import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('boxes.csv')

print("="*80)
print("DATA EXPLORATION")
print("="*80)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

# Research question: How do children's reliance on majority preference develop over growth in age across different cultural contexts?
# Key variables:
# - y: outcome (1=unchosen, 2=majority, 3=minority)
# - age: predictor of interest
# - culture: context variable
# - gender, majority_first: controls

print("\n" + "="*80)
print("UNIVARIATE DISTRIBUTIONS")
print("="*80)

print("\nOutcome distribution (y):")
print(df['y'].value_counts().sort_index())
print("Proportions:")
print(df['y'].value_counts(normalize=True).sort_index())

print("\nAge distribution:")
print(df['age'].value_counts().sort_index())

print("\nCulture distribution:")
print(df['culture'].value_counts().sort_index())

# Create binary variable for whether child chose majority option
df['chose_majority'] = (df['y'] == 2).astype(int)
df['chose_minority'] = (df['y'] == 3).astype(int)
df['chose_unchosen'] = (df['y'] == 1).astype(int)

print("\n" + "="*80)
print("MAJORITY CHOICE BY AGE")
print("="*80)

print("\nProportion choosing majority by age:")
majority_by_age = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print(majority_by_age)

print("\nProportion choosing minority by age:")
minority_by_age = df.groupby('age')['chose_minority'].agg(['mean', 'count'])
print(minority_by_age)

print("\n" + "="*80)
print("STATISTICAL TEST: AGE AND MAJORITY CHOICE")
print("="*80)

# Test 1: Correlation between age and choosing majority
corr_age_majority, p_corr = spearmanr(df['age'], df['chose_majority'])
print(f"\nSpearman correlation between age and choosing majority: r={corr_age_majority:.4f}, p={p_corr:.4f}")

# Test 2: Logistic regression - age predicting majority choice
X_age = sm.add_constant(df['age'])
logit_model_age = sm.Logit(df['chose_majority'], X_age).fit(disp=0)
print("\nLogistic regression: Age -> Majority choice")
print(logit_model_age.summary2().tables[1])

# Test 3: Chi-square test for independence between age groups and outcome
# Create age groups for clearer interpretation
df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 15], labels=['Young (4-6)', 'Middle (7-9)', 'Older (10-14)'])
contingency_table = pd.crosstab(df['age_group'], df['y'])
print("\nContingency table: Age group x Outcome")
print(contingency_table)

chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test: χ²={chi2:.4f}, p={p_chi2:.4f}, df={dof}")

print("\n" + "="*80)
print("CULTURAL CONTEXT ANALYSIS")
print("="*80)

print("\nProportion choosing majority by culture:")
majority_by_culture = df.groupby('culture')['chose_majority'].agg(['mean', 'count'])
print(majority_by_culture)

# Test interaction: age x culture
print("\n" + "="*80)
print("INTERACTION: AGE × CULTURE")
print("="*80)

print("\nProportion choosing majority by age and culture:")
majority_age_culture = df.groupby(['culture', 'age'])['chose_majority'].agg(['mean', 'count'])
print(majority_age_culture)

# Logistic regression with age, culture, and interaction
df['culture_centered'] = df['culture'] - df['culture'].mean()
df['age_centered'] = df['age'] - df['age'].mean()
df['age_culture_interaction'] = df['age_centered'] * df['culture_centered']

X_full = sm.add_constant(df[['age_centered', 'culture_centered', 'age_culture_interaction', 'gender', 'majority_first']])
logit_full = sm.Logit(df['chose_majority'], X_full).fit(disp=0)
print("\nLogistic regression with interaction:")
print(logit_full.summary2().tables[1])

# Test if cultures differ in age effect
print("\n" + "="*80)
print("AGE EFFECTS BY CULTURE")
print("="*80)

culture_age_effects = {}
for culture_id in sorted(df['culture'].unique()):
    df_culture = df[df['culture'] == culture_id]
    if len(df_culture) > 20:  # Only if sufficient data
        corr, p = spearmanr(df_culture['age'], df_culture['chose_majority'])
        culture_age_effects[culture_id] = {'correlation': corr, 'p_value': p, 'n': len(df_culture)}
        print(f"Culture {culture_id}: r={corr:.4f}, p={p:.4f}, n={len(df_culture)}")

print("\n" + "="*80)
print("INTERPRETABLE MODEL: DECISION TREE")
print("="*80)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X_tree = df[['age', 'culture', 'gender', 'majority_first']]
y_tree = df['chose_majority']

dt_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
dt_model.fit(X_tree, y_tree)

print("\nFeature importances:")
for feature, importance in zip(X_tree.columns, dt_model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Synthesize findings
print("\nKey findings:")
print(f"1. Overall correlation between age and majority choice: r={corr_age_majority:.4f}, p={p_corr:.4f}")
print(f"2. Logistic regression age coefficient: {logit_model_age.params['age']:.4f}, p={logit_model_age.pvalues['age']:.4f}")
print(f"3. Chi-square test for age groups: p={p_chi2:.4f}")
print(f"4. Interaction term (age × culture): coef={logit_full.params['age_culture_interaction']:.4f}, p={logit_full.pvalues['age_culture_interaction']:.4f}")

# Determine response based on evidence
# Research question: How do children's reliance on majority preference develop over growth in age across different cultural contexts?

# The question asks about DEVELOPMENT (change with age) across DIFFERENT CULTURAL CONTEXTS
# Key is whether age effects VARY by culture (interaction)

significant_main_age_effect = p_corr < 0.05
significant_interaction = logit_full.pvalues['age_culture_interaction'] < 0.05

# Check if there's variability in age effects across cultures
culture_variations = []
for culture_id, effects in culture_age_effects.items():
    culture_variations.append(effects['correlation'])

variation_in_age_effects = np.std(culture_variations) if len(culture_variations) > 0 else 0

print(f"\nStandard deviation of age-majority correlations across cultures: {variation_in_age_effects:.4f}")

# Determine score
if significant_main_age_effect and significant_interaction:
    # Strong evidence for differential development across cultures
    response = 85
    explanation = f"There is significant evidence that children's reliance on majority preference develops with age (r={corr_age_majority:.3f}, p={p_corr:.4f}), and this development varies significantly across cultural contexts (age×culture interaction p={logit_full.pvalues['age_culture_interaction']:.4f}). Age effects differ across cultures with SD={variation_in_age_effects:.3f}."
elif significant_main_age_effect and variation_in_age_effects > 0.1:
    # Main age effect present with notable cultural variation
    response = 70
    explanation = f"Children's reliance on majority preference increases with age (r={corr_age_majority:.3f}, p={p_corr:.4f}), with notable variation across cultures (SD={variation_in_age_effects:.3f}). The age×culture interaction approaches significance (p={logit_full.pvalues['age_culture_interaction']:.4f})."
elif significant_main_age_effect:
    # Main age effect but limited cultural variation
    response = 50
    explanation = f"While there is a significant age effect on majority preference (r={corr_age_majority:.3f}, p={p_corr:.4f}), the development appears relatively consistent across cultures with limited variation (SD={variation_in_age_effects:.3f}, interaction p={logit_full.pvalues['age_culture_interaction']:.4f})."
else:
    # No significant age effect
    response = 25
    explanation = f"The relationship between age and majority preference is not statistically significant (r={corr_age_majority:.3f}, p={p_corr:.4f}), suggesting limited developmental changes across the age range studied."

print(f"\nFinal response: {response}")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt written successfully")
