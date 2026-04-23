import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('reading.csv')

print("=" * 80)
print("RESEARCH QUESTION:")
print("Does 'Reader View' improve reading speed for individuals with dyslexia?")
print("=" * 80)
print()

# Data overview
print("Dataset shape:", df.shape)
print("\nKey columns:")
print("- reader_view: 0 = off, 1 = on")
print("- dyslexia_bin: 0 = no dyslexia, 1 = has dyslexia")
print("- speed: reading speed (outcome variable)")
print()

# Basic statistics
print("=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
print("\nDistribution of dyslexia:")
print(df['dyslexia_bin'].value_counts())
print("\nDistribution of reader_view:")
print(df['reader_view'].value_counts())
print()

# Filter to only include people with dyslexia
df_dyslexia = df[df['dyslexia_bin'] == 1].copy()
print(f"Number of observations with dyslexia: {len(df_dyslexia)}")
print(f"With reader_view ON: {len(df_dyslexia[df_dyslexia['reader_view'] == 1])}")
print(f"With reader_view OFF: {len(df_dyslexia[df_dyslexia['reader_view'] == 0])}")
print()

# Summary statistics for dyslexic readers
print("\nReading speed statistics for DYSLEXIC readers:")
print(f"Reader View OFF: mean={df_dyslexia[df_dyslexia['reader_view']==0]['speed'].mean():.2f}, "
      f"median={df_dyslexia[df_dyslexia['reader_view']==0]['speed'].median():.2f}, "
      f"std={df_dyslexia[df_dyslexia['reader_view']==0]['speed'].std():.2f}")
print(f"Reader View ON:  mean={df_dyslexia[df_dyslexia['reader_view']==1]['speed'].mean():.2f}, "
      f"median={df_dyslexia[df_dyslexia['reader_view']==1]['speed'].median():.2f}, "
      f"std={df_dyslexia[df_dyslexia['reader_view']==1]['speed'].std():.2f}")
print()

# Check for outliers in speed
print("Speed distribution quantiles (dyslexic readers):")
print(df_dyslexia['speed'].describe())
print()

# Remove extreme outliers in speed (beyond 3 std devs)
mean_speed = df_dyslexia['speed'].mean()
std_speed = df_dyslexia['speed'].std()
df_dyslexia_clean = df_dyslexia[
    (df_dyslexia['speed'] > mean_speed - 3*std_speed) & 
    (df_dyslexia['speed'] < mean_speed + 3*std_speed)
].copy()
print(f"After removing extreme outliers: {len(df_dyslexia_clean)} observations")
print()

# Statistical test: T-test comparing reading speed with and without reader view
speed_with_rv = df_dyslexia_clean[df_dyslexia_clean['reader_view'] == 1]['speed']
speed_without_rv = df_dyslexia_clean[df_dyslexia_clean['reader_view'] == 0]['speed']

print("=" * 80)
print("STATISTICAL TEST: Independent t-test")
print("=" * 80)
print("Comparing reading speed for dyslexic readers:")
print("  H0: Reader View has no effect on reading speed")
print("  H1: Reader View affects reading speed")
print()

t_stat, p_value = stats.ttest_ind(speed_with_rv, speed_without_rv)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")
print()

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(speed_with_rv)-1)*speed_with_rv.std()**2 + 
                       (len(speed_without_rv)-1)*speed_without_rv.std()**2) / 
                      (len(speed_with_rv) + len(speed_without_rv) - 2))
cohens_d = (speed_with_rv.mean() - speed_without_rv.mean()) / pooled_std
print(f"Cohen's d (effect size): {cohens_d:.4f}")
print(f"Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible effect")
elif abs(cohens_d) < 0.5:
    print("small effect")
elif abs(cohens_d) < 0.8:
    print("medium effect")
else:
    print("large effect")
print()

# Regression analysis with controls
print("=" * 80)
print("REGRESSION ANALYSIS (with control variables)")
print("=" * 80)

# Prepare data for regression
df_reg = df_dyslexia_clean.copy()

# Encode categorical variables
le_device = LabelEncoder()
df_reg['device_encoded'] = le_device.fit_transform(df_reg['device'].fillna('unknown'))

le_education = LabelEncoder()
df_reg['education_encoded'] = le_education.fit_transform(df_reg['education'].fillna('unknown'))

# Select features for regression
features = ['reader_view', 'age', 'num_words', 'device_encoded', 
            'education_encoded', 'correct_rate', 'Flesch_Kincaid']
X = df_reg[features].fillna(df_reg[features].mean())
y = df_reg['speed']

# OLS Regression with statsmodels for p-values
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
print()

# Extract coefficient for reader_view
rv_coef = model.params['reader_view']
rv_pval = model.pvalues['reader_view']
rv_conf_int = model.conf_int().loc['reader_view']

print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"Reader View coefficient: {rv_coef:.2f}")
print(f"95% Confidence Interval: [{rv_conf_int[0]:.2f}, {rv_conf_int[1]:.2f}]")
print(f"P-value: {rv_pval:.4f}")
print(f"Statistical significance at α=0.05: {'YES' if rv_pval < 0.05 else 'NO'}")
print()

# Interpretable model: Decision Tree
print("=" * 80)
print("INTERPRETABLE MODEL: Decision Tree")
print("=" * 80)
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
tree.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print("Feature importances:")
print(feature_importance)
print()

# Try imodels if available
try:
    from imodels import HSTreeRegressor
    print("=" * 80)
    print("INTERPRETABLE MODEL: HSTree (Hierarchical Shrinkage Tree)")
    print("=" * 80)
    hstree = HSTreeRegressor(max_leaf_nodes=10, random_state=42)
    hstree.fit(X, y)
    if hasattr(hstree, 'feature_importances_'):
        print("Feature importances:")
        hstree_importance = pd.DataFrame({
            'feature': features,
            'importance': hstree.feature_importances_
        }).sort_values('importance', ascending=False)
        print(hstree_importance)
    else:
        print("HSTree trained successfully (feature importances not available)")
    print()
except (ImportError, AttributeError) as e:
    print("imodels HSTree not available or incompatible, skipping analysis")
    print()

# Mann-Whitney U test (non-parametric alternative)
print("=" * 80)
print("NON-PARAMETRIC TEST: Mann-Whitney U")
print("=" * 80)
u_stat, u_pval = stats.mannwhitneyu(speed_with_rv, speed_without_rv, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {u_pval:.4f}")
print(f"Significant at α=0.05? {'YES' if u_pval < 0.05 else 'NO'}")
print()

# Interaction effect: Does reader view help dyslexia more than non-dyslexia?
print("=" * 80)
print("INTERACTION ANALYSIS")
print("=" * 80)
print("Comparing effect size across dyslexia vs. non-dyslexia:")

# Clean version of full dataset
df_clean = df[
    (df['speed'] > df['speed'].mean() - 3*df['speed'].std()) & 
    (df['speed'] < df['speed'].mean() + 3*df['speed'].std())
].copy()

# Create interaction term
df_clean['interaction'] = df_clean['reader_view'] * df_clean['dyslexia_bin']

# Regression with interaction
X_interaction = df_clean[['reader_view', 'dyslexia_bin', 'interaction']].fillna(0)
y_interaction = df_clean['speed']
X_interaction_const = sm.add_constant(X_interaction)
model_interaction = sm.OLS(y_interaction, X_interaction_const).fit()
print(model_interaction.summary())
print()

interaction_coef = model_interaction.params['interaction']
interaction_pval = model_interaction.pvalues['interaction']
print(f"Interaction coefficient: {interaction_coef:.2f}")
print(f"P-value: {interaction_pval:.4f}")
print(f"Significant interaction at α=0.05? {'YES' if interaction_pval < 0.05 else 'NO'}")
print()

# Final conclusion
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

# Determine score based on statistical tests
if p_value < 0.01 and rv_pval < 0.05:
    if rv_coef > 0:
        score = 85
        explanation = (
            f"Strong evidence that Reader View improves reading speed for dyslexic individuals. "
            f"Independent t-test shows significant difference (p={p_value:.4f}). "
            f"Regression analysis controlling for confounds confirms Reader View increases speed "
            f"by {rv_coef:.1f} units (p={rv_pval:.4f}). Effect size (Cohen's d={cohens_d:.3f}) "
            f"indicates a {'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} "
            f"practical impact."
        )
    else:
        score = 15
        explanation = (
            f"Strong evidence that Reader View actually decreases reading speed for dyslexic individuals. "
            f"Both t-test (p={p_value:.4f}) and regression (p={rv_pval:.4f}) show significant negative effect. "
            f"Reader View decreases speed by {abs(rv_coef):.1f} units."
        )
elif p_value < 0.05:
    if rv_coef > 0:
        score = 70
        explanation = (
            f"Moderate evidence that Reader View improves reading speed for dyslexic individuals. "
            f"T-test shows significant difference (p={p_value:.4f}), though regression p-value is "
            f"marginally significant (p={rv_pval:.4f}). Cohen's d={cohens_d:.3f} suggests "
            f"{'small' if abs(cohens_d) < 0.5 else 'medium'} effect size."
        )
    else:
        score = 30
        explanation = (
            f"Moderate evidence that Reader View may decrease reading speed for dyslexic individuals. "
            f"T-test shows significant difference (p={p_value:.4f}), with negative direction."
        )
elif rv_pval < 0.05:
    if rv_coef > 0:
        score = 65
        explanation = (
            f"Some evidence for improvement. While t-test is not significant (p={p_value:.4f}), "
            f"regression controlling for confounds shows significant positive effect (p={rv_pval:.4f}). "
            f"Reader View increases speed by {rv_coef:.1f} units when accounting for age, education, etc."
        )
    else:
        score = 35
        explanation = (
            f"Mixed evidence. Regression shows significant negative effect (p={rv_pval:.4f}), "
            f"but t-test is not significant (p={p_value:.4f})."
        )
else:
    # Neither test is significant
    score = 45
    explanation = (
        f"Insufficient evidence to conclude Reader View improves reading speed for dyslexic individuals. "
        f"T-test (p={p_value:.4f}) and regression (p={rv_pval:.4f}) both fail to reach statistical "
        f"significance at α=0.05. While descriptive statistics show "
        f"{'higher' if rv_coef > 0 else 'lower'} speed with Reader View, this difference could be due to chance."
    )

print(f"Final Score: {score}/100")
print(f"\nExplanation: {explanation}")
print()

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
