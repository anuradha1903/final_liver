import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("liver.csv", encoding="latin1")

data.columns = [
    'Age',
    'Gender',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'ALT',
    'AST',
    'Total_Proteins',
    'Albumin',
    'Albumin_and_Globulin_Ratio',
    'Result'
]

# ==========================================
# PREPROCESSING
# ==========================================

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# IMPORTANT
# Dataset:
# 1 = Liver Disease
# 2 = No Liver Disease

data['Result'] = data['Result'].replace({
    1: 1,
    2: 0
})

data.fillna(data.mean(numeric_only=True), inplace=True)

X = data.drop("Result", axis=1)
y = data["Result"]

# Save Column Names
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "SVM": SVC(
        kernel='rbf',
        probability=True
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=1,
        colsample_bytree=1,
        random_state=42,
        eval_metric='logloss'
    )
}

accuracies = {}

# ==========================================
# TRAIN ALL MODELS
# ==========================================

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100

    accuracies[name] = accuracy

    print("Accuracy :", round(accuracy, 2), "%")

    # Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        name.replace(" ", "_") + "_CM.png"
    )

    plt.close()

    # ROC Curve

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot([0,1],[0,1],'--')

        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            name.replace(" ","_") + "_ROC.png"
        )

        plt.close()

# ==========================================
# DECISION TREE GRAPH
# ==========================================

plt.figure(figsize=(20,10))

plot_tree(
    models["Decision Tree"],
    filled=True,
    feature_names=data.drop("Result", axis=1).columns,
    class_names=["No Disease","Disease"]
)

plt.savefig("DecisionTree_Graph.png")
plt.close()

# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

rf_importance = models["Random Forest"].feature_importances_

plt.figure(figsize=(10,6))

plt.bar(
    data.drop("Result",axis=1).columns,
    rf_importance
)

plt.xticks(rotation=90)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("RandomForest_FeatureImportance.png")

plt.close()

# ==========================================
# XGBOOST FEATURE IMPORTANCE
# ==========================================

xgb_importance = models["XGBoost"].feature_importances_

plt.figure(figsize=(10,6))

plt.bar(
    data.drop("Result",axis=1).columns,
    xgb_importance
)

plt.xticks(rotation=90)

plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig("XGBoost_FeatureImportance.png")

plt.close()

# ==========================================
# LOGISTIC REGRESSION COEFFICIENTS
# ==========================================

coef = models["Logistic Regression"].coef_[0]

plt.figure(figsize=(10,6))

plt.bar(
    data.drop("Result",axis=1).columns,
    coef
)

plt.xticks(rotation=90)

plt.title("Logistic Regression Coefficients")

plt.tight_layout()

plt.savefig("LogisticRegression_Coefficients.png")

plt.close()

# ==========================================
# SVM SUPPORT VECTORS
# ==========================================

plt.figure(figsize=(6,5))

plt.scatter(
    range(len(models["SVM"].support_)),
    models["SVM"].support_
)

plt.title("SVM Support Vectors")

plt.tight_layout()

plt.savefig("SVM_SupportVectors.png")

plt.close()

# ==========================================
# ACCURACY COMPARISON
# ==========================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    list(accuracies.keys()),
    list(accuracies.values())
)

for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval,
        f"{yval:.2f}%",
        ha='center'
    )

plt.ylabel("Accuracy (%)")
plt.title("All Models Accuracy Comparison")

plt.ylim(0,100)

plt.tight_layout()

plt.savefig("Accuracy_Comparison.png")

plt.close()

# ==========================================
# SAVE BEST MODEL
# ==========================================

pickle.dump(
    models["XGBoost"],
    open("model.pkl", "wb")
)

print("\n✅ ALL MODELS TRAINED")
print("✅ Logistic Regression Visualization Saved")
print("✅ SVM Visualization Saved")
print("✅ Decision Tree Visualization Saved")
print("✅ Random Forest Visualization Saved")
print("✅ XGBoost Visualization Saved")
print("✅ Accuracy Comparison Saved")
print("✅ scaler.pkl Saved")
print("✅ model.pkl Saved")