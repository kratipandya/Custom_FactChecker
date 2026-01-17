import pandas as pd
df = pd.read_csv("llm_ensemble_predictions.csv")
score_col = "ensemble_score" if "ensemble_score" in df.columns else ("llm_score" if "llm_score" in df.columns else None)
print("score_col:", score_col)
print("n rows:", len(df))
print("unique scores (first 20):", df[score_col].unique()[:20])
print(df[score_col].describe())
if "ensemble_label" in df.columns:
    print("\nlabel counts:\n", df["ensemble_label"].value_counts())
elif "llm_label" in df.columns:
    print("\nlabel counts:\n", df["llm_label"].value_counts())
