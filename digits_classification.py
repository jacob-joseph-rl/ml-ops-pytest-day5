import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Model 1: Support Vector Classifier
clf_svm = svm.SVC(gamma=0.001)
clf_svm.fit(X_train, y_train)
predicted_svm = clf_svm.predict(X_test)

# Model 2: Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
predicted_rf = clf_rf.predict(X_test)

# Evaluation for SVM
print(
    f"SVM Classification report:\n"
    f"{metrics.classification_report(y_test, predicted_svm)}\n"
)
disp_svm = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_svm)
disp_svm.figure_.suptitle("SVM Confusion Matrix")
print(f"SVM Confusion matrix:\n{disp_svm.confusion_matrix}")

# Evaluation for Random Forest
print(
    f"Random Forest Classification report:\n"
    f"{metrics.classification_report(y_test, predicted_rf)}\n"
)
disp_rf = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_rf)
disp_rf.figure_.suptitle("Random Forest Confusion Matrix")
print(f"Random Forest Confusion matrix:\n{disp_rf.confusion_matrix}")

plt.show()

# Save the trained models to disk
joblib.dump(clf_svm, "svm_model.joblib")
joblib.dump(clf_rf, "rf_model.joblib")

print("Models saved successfully.")
