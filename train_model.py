import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Drop unnecessary column
df = df.drop(columns=['salary'])

# Convert specialization → Engineering style
df['specialisation'] = df['specialisation'].map({
    'Mkt&Fin': 'Tech',
    'Mkt&HR': 'Non-Tech'
})

# Features & target
X = df.drop('status', axis=1)
y = df['status'].map({'Placed': 1, 'Not Placed': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Columns
categorical_cols = ['gender', 'workex', 'specialisation']
numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# Model
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Pipeline
pipeline = ImbPipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb)
])

# Hyperparameters
param_dist = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__gamma": [0, 0.1, 0.3],
    "model__min_child_weight": [1, 3, 5]
}

# Random Search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Train
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Predict probabilities
y_probs = best_model.predict_proba(X_test)[:, 1]

# Threshold tuning
threshold = 0.4
y_pred = (y_probs >= threshold).astype(int)

# Evaluation
print("\n Best Parameters:", search.best_params_)
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Confusion Matrix (Improved)
cm = confusion_matrix(y_test, y_pred)
labels = ["Not Placed", "Placed"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

ohe = best_model.named_steps['preprocessing'].named_transformers_['cat']
cat_features = ohe.get_feature_names_out(categorical_cols)

all_features = numerical_cols + list(cat_features)

importances = best_model.named_steps['model'].feature_importances_

feat_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Save CSV
feat_df.to_csv("feature_importance.csv", index=False)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(data=feat_df.head(10), x="Importance", y="Feature")

plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()



pickle.dump(best_model, open("best_model.pkl", "wb"))

print("\n Model saved as best_model.pkl")

# Save model
pickle.dump(best_model, open("best_model.pkl", "wb"))

print("\n Model saved as best_model.pkl")