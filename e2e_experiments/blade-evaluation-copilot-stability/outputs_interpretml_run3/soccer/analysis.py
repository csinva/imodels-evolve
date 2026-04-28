import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier

# Load the dataset
df = pd.read_csv('soccer.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Research question: Are soccer players with dark skin tone more likely to receive red cards?
# Key variables: rater1, rater2 (skin tone ratings), redCards (outcome)

# Check for missing values in key columns
print("\nMissing values in key columns:")
print(df[['rater1', 'rater2', 'redCards']].isnull().sum())

# Create average skin tone rating
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Remove rows with missing skin tone data
df_analysis = df.dropna(subset=['skin_tone', 'redCards']).copy()
print(f"\nRows with complete skin tone and red card data: {len(df_analysis)}")

# Summary statistics
print("\nSkin tone distribution:")
print(df_analysis['skin_tone'].describe())
print("\nRed cards distribution:")
print(df_analysis['redCards'].value_counts())

# Create binary indicator for whether player received any red cards
df_analysis['got_red_card'] = (df_analysis['redCards'] > 0).astype(int)
print("\nPlayers who got at least one red card:", df_analysis['got_red_card'].sum())
print("Players who got no red cards:", (df_analysis['got_red_card'] == 0).sum())

# Categorize skin tone: light (0-0.33), medium (0.33-0.67), dark (0.67-1.0)
df_analysis['skin_category'] = pd.cut(df_analysis['skin_tone'], 
                                        bins=[0, 0.33, 0.67, 1.0], 
                                        labels=['light', 'medium', 'dark'],
                                        include_lowest=True)

print("\nSkin category distribution:")
print(df_analysis['skin_category'].value_counts())

# Calculate red card rates by skin tone category
print("\nRed card rates by skin tone category:")
red_card_by_skin = df_analysis.groupby('skin_category')['got_red_card'].agg(['sum', 'count', 'mean'])
red_card_by_skin['rate'] = red_card_by_skin['mean']
print(red_card_by_skin)

# Statistical test: Compare light vs dark skin tone
light_skin = df_analysis[df_analysis['skin_category'] == 'light']['got_red_card']
dark_skin = df_analysis[df_analysis['skin_category'] == 'dark']['got_red_card']

# Chi-square test for independence
contingency_table = pd.crosstab(df_analysis['skin_category'], df_analysis['got_red_card'])
print("\nContingency table:")
print(contingency_table)

chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test: χ² = {chi2:.4f}, p-value = {p_value_chi2:.4e}")

# Two-proportion z-test for light vs dark
if len(light_skin) > 0 and len(dark_skin) > 0:
    light_rate = light_skin.mean()
    dark_rate = dark_skin.mean()
    print(f"\nRed card rate for light skin: {light_rate:.4f}")
    print(f"Red card rate for dark skin: {dark_rate:.4f}")
    print(f"Difference: {dark_rate - light_rate:.4f}")
    
    # Two-sample t-test
    t_stat, p_value_ttest = stats.ttest_ind(dark_skin, light_skin)
    print(f"T-test: t = {t_stat:.4f}, p-value = {p_value_ttest:.4e}")

# Correlation between continuous skin tone and red cards
correlation, p_value_corr = stats.pearsonr(df_analysis['skin_tone'], df_analysis['got_red_card'])
print(f"\nPearson correlation between skin tone and red card: r = {correlation:.4f}, p-value = {p_value_corr:.4e}")

# Spearman correlation (non-parametric)
spearman_corr, p_value_spearman = stats.spearmanr(df_analysis['skin_tone'], df_analysis['got_red_card'])
print(f"Spearman correlation: ρ = {spearman_corr:.4f}, p-value = {p_value_spearman:.4e}")

# Logistic regression with skin tone as predictor
print("\n=== Logistic Regression Analysis ===")
X_simple = df_analysis[['skin_tone']].values
y = df_analysis['got_red_card'].values

# Statsmodels logistic regression for detailed stats
X_sm = sm.add_constant(X_simple)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)
print("\nLogistic Regression Summary:")
print(result.summary())

# Extract key statistics
skin_tone_coef = result.params[1]
skin_tone_pvalue = result.pvalues[1]
odds_ratio = np.exp(skin_tone_coef)
print(f"\nSkin tone coefficient: {skin_tone_coef:.4f}")
print(f"Odds ratio: {odds_ratio:.4f}")
print(f"P-value: {skin_tone_pvalue:.4e}")

# Control for confounders
print("\n=== Controlling for Confounders ===")
# Include games, position, and other factors
df_analysis_full = df_analysis.copy()

# Create dummy variables for categorical variables
position_dummies = pd.get_dummies(df_analysis_full['position'], prefix='pos', drop_first=True)
league_dummies = pd.get_dummies(df_analysis_full['leagueCountry'], prefix='league', drop_first=True)

# Build full model
X_full = pd.concat([
    df_analysis_full[['skin_tone', 'games', 'height', 'weight', 'yellowCards']],
    position_dummies,
    league_dummies
], axis=1)

# Remove any rows with missing values
mask = ~X_full.isnull().any(axis=1)
X_full_clean = X_full[mask]
y_full_clean = df_analysis_full.loc[mask, 'got_red_card'].values

print(f"\nSample size for full model: {len(X_full_clean)}")

# Statsmodels for full model - convert to float64 to avoid dtype issues
X_full_sm = sm.add_constant(X_full_clean.astype(np.float64))
logit_full = sm.Logit(y_full_clean, X_full_sm)
result_full = logit_full.fit(disp=0)

print("\nFull Model - Skin Tone Effect:")
skin_tone_idx = 'skin_tone'  # Use column name directly
print(f"Skin tone coefficient: {result_full.params[skin_tone_idx]:.4f}")
print(f"Odds ratio: {np.exp(result_full.params[skin_tone_idx]):.4f}")
print(f"P-value: {result_full.pvalues[skin_tone_idx]:.4e}")
print(f"95% CI for odds ratio: [{np.exp(result_full.conf_int().loc[skin_tone_idx, 0]):.4f}, {np.exp(result_full.conf_int().loc[skin_tone_idx, 1]):.4f}]")

# Interpretable model: Explainable Boosting Classifier
print("\n=== Explainable Boosting Classifier ===")
ebc = ExplainableBoostingClassifier(random_state=42, max_rounds=1000)
ebc.fit(X_full_clean, y_full_clean)

# Get feature importances from the global explanation
try:
    from interpret import show
    ebm_global = ebc.explain_global()
    feature_names = ebm_global.data()['names']
    importances = ebm_global.data()['scores']
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10))
    
    skin_tone_importance_rank = importance_df[importance_df['feature'] == 'skin_tone'].index[0] + 1 if 'skin_tone' in importance_df['feature'].values else 'N/A'
    print(f"\nSkin tone importance rank: {skin_tone_importance_rank} out of {len(feature_names)}")
except Exception as e:
    print(f"Could not extract feature importances: {e}")
    print("Using simple feature importance estimate from model")

# Conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# Determine response score based on statistical evidence
response_score = 0
explanation_parts = []

# Key evidence:
# 1. Statistical significance
if p_value_corr < 0.05 and correlation > 0:
    explanation_parts.append(f"Significant positive correlation between skin tone and red cards (r={correlation:.3f}, p={p_value_corr:.2e})")
    response_score += 30
elif p_value_corr < 0.05 and correlation < 0:
    explanation_parts.append(f"Significant negative correlation (r={correlation:.3f}, p={p_value_corr:.2e}) - lighter skin associated with more red cards")
    response_score = 5
else:
    explanation_parts.append(f"No significant correlation found (r={correlation:.3f}, p={p_value_corr:.2f})")
    
# 2. Logistic regression significance
if skin_tone_pvalue < 0.05 and skin_tone_coef > 0:
    explanation_parts.append(f"Logistic regression shows significant positive effect (OR={odds_ratio:.3f}, p={skin_tone_pvalue:.2e})")
    response_score += 30
elif skin_tone_pvalue < 0.05 and skin_tone_coef < 0:
    explanation_parts.append(f"Logistic regression shows significant negative effect (OR={odds_ratio:.3f}, p={skin_tone_pvalue:.2e})")
    response_score = 5
else:
    explanation_parts.append(f"Logistic regression not significant (p={skin_tone_pvalue:.2f})")
    
# 3. Full model with controls
if result_full.pvalues[skin_tone_idx] < 0.05 and result_full.params[skin_tone_idx] > 0:
    explanation_parts.append(f"Effect remains significant after controlling for confounders (OR={np.exp(result_full.params[skin_tone_idx]):.3f}, p={result_full.pvalues[skin_tone_idx]:.2e})")
    response_score += 30
elif result_full.pvalues[skin_tone_idx] < 0.05 and result_full.params[skin_tone_idx] < 0:
    explanation_parts.append(f"Effect reverses after controlling for confounders (OR={np.exp(result_full.params[skin_tone_idx]):.3f}, p={result_full.pvalues[skin_tone_idx]:.2e})")
    response_score = 5
else:
    explanation_parts.append(f"Effect not significant after controlling for confounders (p={result_full.pvalues[skin_tone_idx]:.2f})")
    response_score = max(5, response_score // 2)  # Reduce confidence

# 4. Effect size
if len(dark_skin) > 0 and len(light_skin) > 0:
    rate_diff = dark_rate - light_rate
    if rate_diff > 0.01:
        explanation_parts.append(f"Dark-skinned players have {rate_diff*100:.2f}% higher red card rate than light-skinned players")
        response_score += 10
    elif rate_diff < -0.01:
        explanation_parts.append(f"Light-skinned players have {abs(rate_diff)*100:.2f}% higher red card rate")
        response_score = 5

# Cap the score
response_score = min(100, max(0, response_score))

# Adjust based on strength of evidence
if response_score > 50:
    strength = "strong"
elif response_score > 30:
    strength = "moderate"
else:
    strength = "weak or absent"
    
explanation = f"Statistical analysis reveals {strength} evidence that soccer players with darker skin tones are more likely to receive red cards. " + " ".join(explanation_parts[:3])

print(f"\nResponse Score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
