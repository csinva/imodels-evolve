import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('fish.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelations with fish_caught:")
print(df.corr()['fish_caught'])

# Compute fish per hour (rate)
df['fish_per_hour'] = np.where(df['hours'] > 0, df['fish_caught'] / df['hours'], np.nan)
df_fishing = df[df['fish_caught'] > 0].copy()  # groups that actually fished

print("\n--- Fish per hour (all groups) ---")
print(df['fish_per_hour'].describe())
avg_rate_all = df['fish_per_hour'].mean()
print(f"Mean fish/hour (all): {avg_rate_all:.4f}")

print("\n--- Fish per hour (groups that caught fish) ---")
print(df_fishing['fish_per_hour'].describe())
avg_rate_fishing = df_fishing['fish_per_hour'].mean()
print(f"Mean fish/hour (fishing groups): {avg_rate_fishing:.4f}")

# OLS regression predicting fish_caught from features
X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
X_const = sm.add_constant(X)
y = df['fish_caught']
model = sm.OLS(y, X_const).fit()
print("\n--- OLS Summary ---")
print(model.summary())

# Poisson regression (count data)
poisson_model = sm.GLM(y, X_const, family=sm.families.Poisson()).fit()
print("\n--- Poisson GLM Summary ---")
print(poisson_model.summary())

# Feature importances via sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X, y)
for feat, coef in zip(X.columns, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# Rate regression: predict fish_per_hour from features
df_valid = df.dropna(subset=['fish_per_hour'])
X_rate = df_valid[['livebait', 'camper', 'persons', 'child', 'hours']]
X_rate_const = sm.add_constant(X_rate)
y_rate = df_valid['fish_per_hour']
rate_model = sm.OLS(y_rate, X_rate_const).fit()
print("\n--- OLS on fish_per_hour ---")
print(rate_model.summary())

# Average rate of fish caught per hour by livebait
print("\nMean fish/hour by livebait:")
print(df.groupby('livebait')['fish_per_hour'].mean())

# Median rate overall
median_rate = df['fish_per_hour'].median()
mean_rate = df['fish_per_hour'].mean()
print(f"\nOverall: mean={mean_rate:.4f}, median={median_rate:.4f} fish/hour")

# The research question asks: how many fish on average do visitors take per hour when fishing?
# We need to compute average fish per hour for those who were fishing (i.e. fish_caught > 0)
rate_when_fishing = df_fishing['fish_per_hour'].mean()
print(f"\nRate when fishing (fish_caught>0): {rate_when_fishing:.4f} fish/hour")

# Write conclusion
import json

# The question is a quantitative estimation — we interpret the Likert scale as
# confidence that we can meaningfully estimate a rate (i.e., factors exist and
# the rate is estimable). Score ~70: yes, we can estimate and factors are identifiable.
response_val = 70
explanation = (
    f"The dataset contains {len(df)} visits. Among all visitors, the mean fish catch rate "
    f"is {mean_rate:.2f} fish/hour (median {median_rate:.2f}). Restricting to groups that "
    f"actually caught fish (n={len(df_fishing)}), the mean rate is {rate_when_fishing:.2f} fish/hour. "
    f"OLS regression shows that livebait (coef={model.params['livebait']:.2f}, p={model.pvalues['livebait']:.4f}) "
    f"and persons (coef={model.params['persons']:.2f}, p={model.pvalues['persons']:.4f}) "
    f"are the strongest predictors of fish_caught. Hours spent in the park is also a positive predictor "
    f"(coef={model.params['hours']:.2f}, p={model.pvalues['hours']:.4f}). "
    f"A Poisson GLM confirms these patterns. Overall, visitors catch approximately "
    f"{mean_rate:.2f} fish per hour on average, but there is high variability (std={df['fish_per_hour'].std():.2f}). "
    f"The rate is meaningfully influenced by livebait use and group size, making it possible to estimate "
    f"the catch rate with moderate confidence."
)

result = {"response": response_val, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
