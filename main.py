# ============================================================
# IPL WINNING TEAM PROBABILITY PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("IPL_Matches_2022.csv")

print("Dataset Shape:", df.shape)

print("\nColumns in dataset:")
print(df.columns.tolist())

# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Team1",
    "Team2",
    "WinningTeam"
]

for col in required_columns:
    if col not in df.columns:
        print(f"ERROR: Column '{col}' not found!")

# ============================================================
# 4. REMOVE MISSING WINNERS
# ============================================================

df = df.dropna(subset=["WinningTeam"])

print("\nData after removing missing winners:")
print(df.shape)

# ============================================================
# 5. CREATE TARGET
# ============================================================

# Target:
# 1 = Team1 wins
# 0 = Team2 wins

df["target"] = np.where(
    df["Team1"] == df["WinningTeam"],
    1,
    0
)

print("\nTarget distribution:")
print(df["target"].value_counts())

# ============================================================
# 6. HISTORICAL TEAM STATISTICS
# ============================================================

team_stats = {}


def get_team_stats(team):

    if team not in team_stats:
        team_stats[team] = {
            "matches": 0,
            "wins": 0
        }

    matches = team_stats[team]["matches"]
    wins = team_stats[team]["wins"]

    if matches == 0:
        win_rate = 0.50
    else:
        win_rate = wins / matches

    return matches, wins, win_rate


# ============================================================
# 7. CREATE FEATURES
# ============================================================

features = []

# Keep chronological order
if "ID" in df.columns:
    df = df.sort_values("ID")

for _, row in df.iterrows():

    team1 = row["Team1"]
    team2 = row["Team2"]

    # Historical statistics BEFORE the match
    t1_matches, t1_wins, t1_winrate = get_team_stats(team1)
    t2_matches, t2_wins, t2_winrate = get_team_stats(team2)

    features.append({

        "Team1": team1,
        "Team2": team2,

        "Team1_Matches": t1_matches,
        "Team1_Wins": t1_wins,
        "Team1_WinRate": t1_winrate,

        "Team2_Matches": t2_matches,
        "Team2_Wins": t2_wins,
        "Team2_WinRate": t2_winrate,

        "WinRate_Difference":
            t1_winrate - t2_winrate,

        "target": row["target"]
    })

    # Update statistics AFTER the match
    team_stats.setdefault(
        team1,
        {"matches": 0, "wins": 0}
    )

    team_stats.setdefault(
        team2,
        {"matches": 0, "wins": 0}
    )

    team_stats[team1]["matches"] += 1
    team_stats[team2]["matches"] += 1

    if row["WinningTeam"] == team1:

        team_stats[team1]["wins"] += 1

    elif row["WinningTeam"] == team2:

        team_stats[team2]["wins"] += 1


model_df = pd.DataFrame(features)

print("\nModel Dataset:")
print(model_df.head())

# ============================================================
# 8. FEATURES AND TARGET
# ============================================================

X = model_df.drop(columns=["target"])

y = model_df["target"]

categorical_features = [
    "Team1",
    "Team2"
]

numeric_features = [
    "Team1_Matches",
    "Team1_Wins",
    "Team1_WinRate",
    "Team2_Matches",
    "Team2_Wins",
    "Team2_WinRate",
    "WinRate_Difference"
]

# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ============================================================
# 10. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(model_df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

# ============================================================
# 11. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=8,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced"
)

pipeline = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        ("model", rf_model)
    ]
)

# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")

# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

if len(np.unique(y_test)) == 2:

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        "\nROC-AUC:",
        round(auc, 4)
    )

loss = log_loss(
    y_test,
    y_probability
)

print(
    "Log Loss:",
    round(loss, 4)
)

# ============================================================
# 15. PREDICT IPL MATCH
# ============================================================

def predict_match(team1, team2):

    t1_matches, t1_wins, t1_winrate = \
        get_team_stats(team1)

    t2_matches, t2_wins, t2_winrate = \
        get_team_stats(team2)

    input_data = pd.DataFrame([{

        "Team1": team1,

        "Team2": team2,

        "Team1_Matches": t1_matches,

        "Team1_Wins": t1_wins,

        "Team1_WinRate": t1_winrate,

        "Team2_Matches": t2_matches,

        "Team2_Wins": t2_wins,

        "Team2_WinRate": t2_winrate,

        "WinRate_Difference":
            t1_winrate - t2_winrate
    }])

    probability = pipeline.predict_proba(
        input_data
    )[0]

    team1_probability = probability[1]

    team2_probability = probability[0]

    print("\n================================")
    print("IPL WINNING PROBABILITY")
    print("================================")

    print(
        f"{team1}: "
        f"{team1_probability * 100:.2f}%"
    )

    print(
        f"{team2}: "
        f"{team2_probability * 100:.2f}%"
    )

    if team1_probability > team2_probability:

        print(
            f"\nPredicted Winner: {team1}"
        )

    else:

        print(
            f"\nPredicted Winner: {team2}"
        )


# ============================================================
# 16. EXAMPLE
# ============================================================

predict_match(
    "Chennai Super Kings",
    "Mumbai Indians"
)

# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pipeline.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 10 Important Features:")

print(
    importance_df.head(10)
)

# ============================================================
# 18. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance_df.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 10 Features - IPL Win Prediction"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()