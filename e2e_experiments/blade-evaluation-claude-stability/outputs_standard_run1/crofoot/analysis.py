import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('crofoot.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df['win'].mean())

# Derived features
df['rel_size'] = df['n_focal'] / df['n_other']          # relative group size
df['rel_dist'] = df['dist_focal'] / df['dist_other']    # focal closer to home = lower value

# Contest location: "home" advantage measured by relative distance
# If focal is closer to its home range center than the other group is to theirs,
# focal has "home advantage" (rel_dist < 1)
df['location_advantage'] = df['dist_other'] - df['dist_focal']  # positive = focal has home adv

# Summary split by win
print("\nWin=1 stats:")
print(df[df['win']==1][['rel_size','rel_dist','location_advantage']].describe())
print("\nWin=0 stats:")
print(df[df['win']==0][['rel_size','rel_dist','location_advantage']].describe())

# T-tests
t_size, p_size = stats.ttest_ind(df[df['win']==1]['rel_size'], df[df['win']==0]['rel_size'])
t_loc, p_loc   = stats.ttest_ind(df[df['win']==1]['location_advantage'], df[df['win']==0]['location_advantage'])
print(f"\nRelative group size: t={t_size:.3f}, p={p_size:.4f}")
print(f"Location advantage:  t={t_loc:.3f}, p={p_loc:.4f}")

# Logistic regression with statsmodels for p-values
X = df[['rel_size', 'location_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X).fit(disp=0)
print("\n", logit_model.summary())

p_size_logit = logit_model.pvalues['rel_size']
p_loc_logit  = logit_model.pvalues['location_advantage']
coef_size    = logit_model.params['rel_size']
coef_loc     = logit_model.params['location_advantage']

print(f"\nLogistic reg p-value (rel_size): {p_size_logit:.4f}")
print(f"Logistic reg p-value (location_advantage): {p_loc_logit:.4f}")

# Determine response score
# Both significant → strong yes (~85-95)
# One significant → moderate yes (~65-75)
# Neither significant → no (~20-35)

size_sig = p_size_logit < 0.05
loc_sig  = p_loc_logit  < 0.05

if size_sig and loc_sig:
    response = 88
    explanation = (
        f"Both relative group size (logistic regression coef={coef_size:.3f}, p={p_size_logit:.4f}) "
        f"and contest location / home-range distance advantage (coef={coef_loc:.3f}, p={p_loc_logit:.4f}) "
        "are statistically significant predictors of winning an intergroup contest. "
        "Larger focal groups and groups contesting closer to their own home range center "
        "(i.e., farther from the rival's home range) have a significantly higher probability of winning."
    )
elif size_sig:
    response = 65
    explanation = (
        f"Relative group size (p={p_size_logit:.4f}) significantly predicts contest outcome, "
        f"but contest location advantage is not significant (p={p_loc_logit:.4f}). "
        "Group size is confirmed as a predictor, but the home-range location effect is weak in this sample."
    )
elif loc_sig:
    response = 65
    explanation = (
        f"Contest location (home-range distance advantage, p={p_loc_logit:.4f}) significantly predicts "
        f"contest outcome, but relative group size is not significant (p={p_size_logit:.4f}). "
        "Location matters, but group size alone is not a reliable predictor in this dataset."
    )
else:
    response = 25
    explanation = (
        f"Neither relative group size (p={p_size_logit:.4f}) nor contest location advantage "
        f"(p={p_loc_logit:.4f}) reaches statistical significance in logistic regression. "
        "The data do not strongly support either factor as a reliable predictor of contest outcome."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
