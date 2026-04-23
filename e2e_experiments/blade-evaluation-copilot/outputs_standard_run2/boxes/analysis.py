import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor
import json

# Load the data
df = pd.read_csv('boxes.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head(10))
print("\nSummary statistics:")
print(df.describe())
print("\nValue counts for outcome (y):")
print(df['y'].value_counts().sort_index())
print("\n1=unchosen option, 2=majority option, 3=minority option")

# Create binary indicator for choosing majority option
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\n" + "=" * 80)
print("RESEARCH QUESTION ANALYSIS")
print("=" * 80)
print("How do children's reliance on majority preference develop over growth in age")
print("across different cultural contexts?")

# Overall majority choice rate by age
print("\n\nMajority choice rate by age:")
majority_by_age = df.groupby('age')['chose_majority'].agg(['mean', 'count'])
print(majority_by_age)

# Majority choice rate by culture
print("\n\nMajority choice rate by culture:")
majority_by_culture = df.groupby('culture')['chose_majority'].agg(['mean', 'count'])
print(majority_by_culture)

# Test 1: Correlation between age and choosing majority
print("\n" + "=" * 80)
print("TEST 1: CORRELATION BETWEEN AGE AND MAJORITY CHOICE")
print("=" * 80)
correlation_result = stats.pearsonr(df['age'], df['chose_majority'])
print(f"Pearson correlation: r = {correlation_result[0]:.4f}, p = {correlation_result[1]:.4f}")

# Test 2: Logistic regression - age predicting majority choice
print("\n" + "=" * 80)
print("TEST 2: LOGISTIC REGRESSION - AGE PREDICTING MAJORITY CHOICE")
print("=" * 80)
X_age = sm.add_constant(df['age'])
logit_model = sm.Logit(df['chose_majority'], X_age)
logit_result = logit_model.fit(disp=0)
print(logit_result.summary())

# Test 3: Does the age effect vary by culture? (Age x Culture interaction)
print("\n" + "=" * 80)
print("TEST 3: AGE x CULTURE INTERACTION")
print("=" * 80)

# Create interaction term
df['age_x_culture'] = df['age'] * df['culture']

# Logistic regression with interaction
X_interaction = sm.add_constant(df[['age', 'culture', 'age_x_culture']])
logit_interaction = sm.Logit(df['chose_majority'], X_interaction)
interaction_result = logit_interaction.fit(disp=0)
print(interaction_result.summary())

# Test 4: Linear regression for each culture separately
print("\n" + "=" * 80)
print("TEST 4: AGE EFFECT BY CULTURE (SEPARATE REGRESSIONS)")
print("=" * 80)
culture_results = {}
for culture_id in sorted(df['culture'].unique()):
    culture_data = df[df['culture'] == culture_id]
    if len(culture_data) > 10:  # Only analyze cultures with sufficient data
        X_culture = sm.add_constant(culture_data['age'])
        model = sm.Logit(culture_data['chose_majority'], X_culture)
        try:
            result = model.fit(disp=0)
            age_coef = result.params['age']
            age_pval = result.pvalues['age']
            culture_results[culture_id] = {
                'coef': age_coef,
                'pval': age_pval,
                'n': len(culture_data),
                'mean_majority': culture_data['chose_majority'].mean()
            }
            print(f"\nCulture {culture_id} (n={len(culture_data)}): "
                  f"Age coef={age_coef:.4f}, p={age_pval:.4f}")
        except:
            print(f"\nCulture {culture_id}: Model did not converge")

# Test 5: Interpretable model - Decision tree
print("\n" + "=" * 80)
print("TEST 5: INTERPRETABLE MODELS")
print("=" * 80)

# Try FIGS regressor for interpretability
X_features = df[['age', 'culture', 'gender', 'majority_first']].values
y_target = df['chose_majority'].values

try:
    figs = FIGSRegressor(max_rules=10)
    figs.fit(X_features, y_target)
    print("\nFIGS Model Rules:")
    print(figs)
except Exception as e:
    print(f"FIGS model failed: {e}")

# Try HSTree as alternative
try:
    hstree = HSTreeRegressor(max_leaf_nodes=8)
    hstree.fit(X_features, y_target)
    print("\nHSTree Model:")
    print(hstree)
except Exception as e:
    print(f"HSTree model failed: {e}")

# Test 6: ANOVA - is there a significant difference in majority choice across age groups?
print("\n" + "=" * 80)
print("TEST 6: ANOVA - AGE GROUPS")
print("=" * 80)

# Create age groups
df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 15], labels=['Young (4-6)', 'Middle (7-9)', 'Old (10-14)'])
age_groups = [group['chose_majority'].values for name, group in df.groupby('age_group', observed=True)]
f_stat, p_value = stats.f_oneway(*age_groups)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

print("\nMajority choice rate by age group:")
print(df.groupby('age_group', observed=True)['chose_majority'].agg(['mean', 'count']))

# Test 7: Chi-square test for each culture
print("\n" + "=" * 80)
print("TEST 7: CHI-SQUARE TESTS BY CULTURE")
print("=" * 80)
for culture_id in sorted(df['culture'].unique()):
    culture_data = df[df['culture'] == culture_id]
    if len(culture_data) > 10:
        # Create age groups within culture
        culture_data_copy = culture_data.copy()
        culture_data_copy['age_cat'] = pd.cut(culture_data_copy['age'], bins=[3, 7, 11, 15], labels=['Young', 'Middle', 'Old'])
        contingency = pd.crosstab(culture_data_copy['age_cat'], culture_data_copy['chose_majority'])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            print(f"Culture {culture_id}: χ²={chi2:.4f}, p={p:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Synthesize findings
overall_age_sig = correlation_result[1] < 0.05
overall_age_positive = correlation_result[0] > 0
logit_age_sig = logit_result.pvalues['age'] < 0.05
interaction_sig = interaction_result.pvalues['age_x_culture'] < 0.05

# Count how many cultures show positive age effect
positive_cultures = sum(1 for c in culture_results.values() if c['coef'] > 0 and c['pval'] < 0.05)
total_cultures = len(culture_results)

print(f"\nKey Findings:")
print(f"1. Overall correlation between age and majority choice: r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}")
print(f"2. Logistic regression age coefficient: {logit_result.params['age']:.4f}, p={logit_result.pvalues['age']:.4f}")
print(f"3. Age x Culture interaction p-value: {interaction_result.pvalues['age_x_culture']:.4f}")
print(f"4. Cultures with significant positive age effect: {positive_cultures}/{total_cultures}")

# Decision logic
if overall_age_sig and overall_age_positive:
    if interaction_sig:
        # There is an age effect, but it varies significantly by culture
        if positive_cultures >= total_cultures * 0.5:
            response = 70
            explanation = (f"There is a significant positive relationship between age and reliance on majority "
                          f"preference (r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}), "
                          f"but the strength of this relationship varies significantly across cultures "
                          f"(interaction p={interaction_result.pvalues['age_x_culture']:.4f}). "
                          f"{positive_cultures} out of {total_cultures} cultures show significant positive age effects.")
        else:
            response = 60
            explanation = (f"While there is an overall positive relationship between age and majority preference "
                          f"(r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}), the age x culture interaction "
                          f"is significant (p={interaction_result.pvalues['age_x_culture']:.4f}), suggesting considerable "
                          f"variation across cultural contexts. Only {positive_cultures}/{total_cultures} cultures show "
                          f"significant positive age effects.")
    else:
        # Strong age effect, consistent across cultures
        response = 85
        explanation = (f"There is a strong and consistent relationship between age and reliance on majority preference. "
                      f"The correlation is significant (r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}), "
                      f"and the age x culture interaction is not significant (p={interaction_result.pvalues['age_x_culture']:.4f}), "
                      f"indicating that the developmental trajectory is similar across cultures.")
else:
    if interaction_sig:
        # No overall effect, but culture-specific patterns exist
        if positive_cultures > 0:
            response = 50
            explanation = (f"The overall relationship between age and majority preference is not significant "
                          f"(r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}), but there is a significant "
                          f"age x culture interaction (p={interaction_result.pvalues['age_x_culture']:.4f}). "
                          f"This suggests culture-specific developmental patterns, with {positive_cultures}/{total_cultures} "
                          f"cultures showing significant positive effects.")
        else:
            response = 30
            explanation = (f"The relationship between age and majority preference varies significantly by culture "
                          f"(interaction p={interaction_result.pvalues['age_x_culture']:.4f}), but no clear overall pattern "
                          f"emerges (overall r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}).")
    else:
        # No significant effect at all
        response = 15
        explanation = (f"There is no significant relationship between age and reliance on majority preference "
                      f"(r={correlation_result[0]:.3f}, p={correlation_result[1]:.4f}). The age x culture interaction "
                      f"is also not significant (p={interaction_result.pvalues['age_x_culture']:.4f}), suggesting that "
                      f"children's majority preference does not systematically develop with age across cultures.")

print(f"\nResponse Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
result = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
