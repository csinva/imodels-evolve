import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')

print("="*80)
print("RESEARCH QUESTION:")
print("Are soccer players with a dark skin tone more likely than those with")
print("a light skin tone to receive red cards from referees?")
print("="*80)

# Explore the data
print("\nDATA OVERVIEW:")
print(f"Total observations: {len(df)}")
print(f"Number of unique players: {df['playerShort'].nunique()}")
print(f"Number of unique referees: {df['refNum'].nunique()}")

# Examine skin tone ratings
print("\n" + "="*80)
print("SKIN TONE ANALYSIS:")
print("="*80)
# Average the two raters to get overall skin tone
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2

# Remove rows where skin tone is missing
df_with_skin = df[df['skin_tone'].notna()].copy()
print(f"\nObservations with skin tone data: {len(df_with_skin)} out of {len(df)}")

print("\nSkin tone distribution:")
print(df_with_skin['skin_tone'].describe())

# Create categories for interpretation
df_with_skin['skin_category'] = pd.cut(df_with_skin['skin_tone'], 
                                         bins=[0, 0.25, 0.75, 1.0],
                                         labels=['Light', 'Medium', 'Dark'])

print("\nSkin tone categories:")
print(df_with_skin['skin_category'].value_counts())

# Examine red card distribution
print("\n" + "="*80)
print("RED CARD ANALYSIS:")
print("="*80)
print(f"\nTotal red cards issued: {df_with_skin['redCards'].sum()}")
print(f"Dyads with at least one red card: {(df_with_skin['redCards'] > 0).sum()}")
print(f"Red card rate: {df_with_skin['redCards'].mean():.4f} per dyad")

# Red cards by skin tone category
print("\nRed cards by skin tone category:")
for cat in ['Light', 'Medium', 'Dark']:
    cat_data = df_with_skin[df_with_skin['skin_category'] == cat]
    if len(cat_data) > 0:
        red_card_rate = cat_data['redCards'].mean()
        red_cards_total = cat_data['redCards'].sum()
        print(f"  {cat}: {red_card_rate:.5f} per dyad ({red_cards_total} total red cards, {len(cat_data)} dyads)")

# Statistical test: Compare light vs dark skin tones
print("\n" + "="*80)
print("STATISTICAL TESTS:")
print("="*80)

# Create binary categories: light (skin_tone < 0.5) vs dark (skin_tone >= 0.5)
df_with_skin['is_dark_skin'] = (df_with_skin['skin_tone'] >= 0.5).astype(int)
light_skin = df_with_skin[df_with_skin['is_dark_skin'] == 0]
dark_skin = df_with_skin[df_with_skin['is_dark_skin'] == 1]

print(f"\nLight skin players (skin_tone < 0.5): {len(light_skin)} dyads")
print(f"  Red cards: {light_skin['redCards'].sum()}, Rate: {light_skin['redCards'].mean():.5f}")
print(f"\nDark skin players (skin_tone >= 0.5): {len(dark_skin)} dyads")
print(f"  Red cards: {dark_skin['redCards'].sum()}, Rate: {dark_skin['redCards'].mean():.5f}")

# T-test comparing red card rates
t_stat, p_value_ttest = stats.ttest_ind(dark_skin['redCards'], light_skin['redCards'])
print(f"\nT-test (dark vs light skin):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4e}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mann = stats.mannwhitneyu(dark_skin['redCards'], light_skin['redCards'], alternative='greater')
print(f"\nMann-Whitney U test (dark > light):")
print(f"  U-statistic: {u_stat:.2f}")
print(f"  p-value: {p_value_mann:.4e}")

# Logistic regression: Predict whether a red card was given based on skin tone
print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL:")
print("="*80)

# Create binary outcome: any red card
df_with_skin['has_red_card'] = (df_with_skin['redCards'] > 0).astype(int)

# Simple model: skin tone only
X_simple = df_with_skin[['skin_tone']].values
y = df_with_skin['has_red_card'].values

# Fit using statsmodels for p-values
X_simple_sm = sm.add_constant(X_simple)
logit_simple = sm.Logit(y, X_simple_sm)
result_simple = logit_simple.fit(disp=0)

print("\nSimple model (skin tone only):")
print(result_simple.summary2().tables[1])

# Control for confounders
print("\n" + "="*80)
print("CONTROLLED ANALYSIS:")
print("="*80)

# Include control variables that might affect red card likelihood
control_vars = ['games', 'yellowCards', 'position']

# Prepare data
df_control = df_with_skin.dropna(subset=['games', 'yellowCards', 'position']).copy()

# One-hot encode position
position_dummies = pd.get_dummies(df_control['position'], prefix='pos', drop_first=True)
df_control = pd.concat([df_control, position_dummies], axis=1)

# Create feature matrix
feature_cols = ['skin_tone', 'games', 'yellowCards'] + list(position_dummies.columns)
X_control = df_control[feature_cols].fillna(0)
y_control = df_control['has_red_card'].values

# Standardize continuous features
scaler = StandardScaler()
X_control_scaled = X_control.copy()
X_control_scaled[['skin_tone', 'games', 'yellowCards']] = scaler.fit_transform(
    X_control[['skin_tone', 'games', 'yellowCards']]
)

# Convert to numpy array to avoid pandas dtype issues
X_control_array = X_control_scaled.values.astype(float)

# Fit model
X_control_sm = sm.add_constant(X_control_array)
logit_control = sm.Logit(y_control, X_control_sm)
result_control = logit_control.fit(disp=0)

print("\nControlled model (with games, yellow cards, position):")
print(result_control.summary2().tables[1])

# Extract skin tone coefficient and p-value (skin_tone is the first feature after constant)
skin_tone_coef = result_control.params[1]
skin_tone_pval = result_control.pvalues[1]

print(f"\n\nSkin tone coefficient: {skin_tone_coef:.4f}")
print(f"Skin tone p-value: {skin_tone_pval:.4e}")
print(f"Odds ratio: {np.exp(skin_tone_coef):.4f}")

# Effect size interpretation
print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)

diff_rate = dark_skin['redCards'].mean() - light_skin['redCards'].mean()
percent_increase = (dark_skin['redCards'].mean() / light_skin['redCards'].mean() - 1) * 100

print(f"\nDark skin players receive red cards at a rate {percent_increase:.1f}% higher than light skin players")
print(f"Absolute difference: {diff_rate:.5f} red cards per dyad")

# Decision criteria
alpha = 0.05
is_significant_ttest = p_value_ttest < alpha
is_significant_mann = p_value_mann < alpha
is_significant_logit_simple = result_simple.pvalues[1] < alpha
is_significant_logit = skin_tone_pval < alpha

print(f"\nStatistical significance (α = {alpha}):")
print(f"  T-test: {'YES' if is_significant_ttest else 'NO'} (p = {p_value_ttest:.4e})")
print(f"  Mann-Whitney U: {'YES' if is_significant_mann else 'NO'} (p = {p_value_mann:.4e})")
print(f"  Logistic regression (simple): {'YES' if is_significant_logit_simple else 'NO'} (p = {result_simple.pvalues[1]:.4e})")
print(f"  Logistic regression (controlled): {'YES' if is_significant_logit else 'NO'} (p = {skin_tone_pval:.4e})")

# Final conclusion
print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

# The controlled logistic regression is the most robust test
if is_significant_logit:
    if is_significant_logit_simple and is_significant_mann:
        response_score = 85
        explanation = (
            f"Yes, the data provides strong evidence that dark-skinned players receive more red cards. "
            f"Dark-skinned players received red cards at a rate {percent_increase:.1f}% higher than "
            f"light-skinned players. This relationship is statistically significant in the controlled "
            f"logistic regression analysis (p={skin_tone_pval:.4e}, OR={np.exp(skin_tone_coef):.3f}), "
            f"which accounts for confounders like number of games and yellow cards. The simple logistic "
            f"regression (p={result_simple.pvalues[1]:.4e}) and Mann-Whitney test (p={p_value_mann:.4e}) "
            f"also support this finding."
        )
    else:
        response_score = 70
        explanation = (
            f"Yes, the data suggests that dark-skinned players receive more red cards. "
            f"Dark-skinned players received red cards at a rate {percent_increase:.1f}% higher. "
            f"The controlled logistic regression shows a significant relationship (p={skin_tone_pval:.4e}, "
            f"OR={np.exp(skin_tone_coef):.3f}) even when accounting for confounders. While simpler "
            f"tests (t-test p={p_value_ttest:.4e}, Mann-Whitney p={p_value_mann:.4e}) show marginal "
            f"or non-significant results, the more sophisticated controlled analysis provides evidence "
            f"for the relationship."
        )
elif is_significant_logit_simple:
    response_score = 55
    explanation = (
        f"There is mixed evidence regarding whether dark-skinned players receive more red cards. "
        f"The simple logistic regression shows significance (p={result_simple.pvalues[1]:.4e}), "
        f"but when controlling for confounders, the relationship becomes less clear (p={skin_tone_pval:.4e}). "
        f"Dark-skinned players had a {percent_increase:.1f}% higher rate, but this may be partly "
        f"explained by other factors."
    )
else:
    response_score = 25
    explanation = (
        f"No, the data does not show a statistically significant relationship between skin tone "
        f"and red card likelihood. While dark-skinned players had a {percent_increase:.1f}% "
        f"higher rate, this difference was not significant across multiple statistical tests "
        f"(t-test p={p_value_ttest:.4e}, Mann-Whitney p={p_value_mann:.4e})."
    )

print(f"\nResponse Score: {response_score}/100")
print(f"\n{explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete. Results written to conclusion.txt")
print("="*80)
