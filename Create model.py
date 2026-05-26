import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DATA_FILE = "fall_training_features_2.csv"
MODEL_FILE = "fall_detection_model.pkl"

LABELS = [
    "walking",
    "fall",
    "stand_to_sit",
    "sit_to_stand",
    "bending"
]

FEATURE_COLUMNS = [
    "ax_mean", "ay_mean", "az_mean",
    "gx_mean", "gy_mean", "gz_mean",

    "ax_std", "ay_std", "az_std",
    "gx_std", "gy_std", "gz_std",

    "ax_min", "ay_min", "az_min",
    "gx_min", "gy_min", "gz_min",

    "ax_max", "ay_max", "az_max",
    "gx_max", "gy_max", "gz_max",

    "acc_mag_mean", "acc_mag_std",
    "acc_mag_min", "acc_mag_max",

    "gyro_mag_mean", "gyro_mag_std",
    "gyro_mag_min", "gyro_mag_max",

    "acc_energy", "gyro_energy",

    "sample_count"
]


print("==============================================")
print("🧠 Fall Detection Model Training")
print("==============================================")

if not os.path.exists(DATA_FILE):
    print(f"❌ Training data олдсонгүй: {DATA_FILE}")
    print("Эхлээд collect_fall_data.py ажиллуулж data цуглуулна уу.")
    exit()

df = pd.read_csv(DATA_FILE)

print(f"📄 Data file: {DATA_FILE}")
print(f"📊 Total rows: {len(df)}")

required_columns = ["label"] + FEATURE_COLUMNS
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("❌ Дараах column-ууд байхгүй байна:")
    for col in missing_columns:
        print(f" - {col}")
    exit()

df = df[df["label"].isin(LABELS)].copy()

if len(df) == 0:
    print("❌ Ашиглах боломжтой label бүхий data алга.")
    exit()

df = df.dropna(subset=["label"])

print("\n🏷 Label distribution:")
print(df["label"].value_counts())

X = df[FEATURE_COLUMNS]
y = df["label"]

if len(df) < 5:
    print("\n⚠️ Data хэт бага байна. Гэхдээ model train хийж хадгална.")
    use_test_split = False
else:
    label_counts = y.value_counts()
    min_count = label_counts.min()

    if min_count >= 2 and len(label_counts) >= 2:
        use_test_split = True
    else:
        use_test_split = False
        print("\n⚠️ Зарим label дээр 1 л sample байна. Test split хийхгүй.")


model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    ))
])

if use_test_split:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\n⏳ Model train хийж байна...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n✅ Training дууслаа.")
    print("==============================================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

else:
    print("\n⏳ Бүх data дээр model train хийж байна...")
    model.fit(X, y)
    print("✅ Training дууслаа.")
    print("⚠️ Test evaluation хийгдээгүй. Илүү их data цуглуулаарай.")

model_package = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "labels": LABELS,
    "record_seconds": 5.0,
    "acc_delta_threshold": 0.20,
    "gyro_threshold": 15.0
}

joblib.dump(model_package, MODEL_FILE)

print("==============================================")
print(f"💾 Model хадгалагдлаа: {MODEL_FILE}")
print("==============================================")
