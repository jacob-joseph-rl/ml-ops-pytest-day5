import os
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

def test_model_files_exist():
    assert os.path.exists("svm_model.joblib"), "SVM model file does not exist!"
    assert os.path.exists("rf_model.joblib"), "Random Forest model file does not exist!"

def test_loaded_models_performance():
    clf_svm = joblib.load("svm_model.joblib")
    clf_rf = joblib.load("rf_model.joblib")

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    _, X_test, _, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    predicted_svm = clf_svm.predict(X_test)
    predicted_rf = clf_rf.predict(X_test)

    # Check if accuracy is above a threshold to ensure models predict reasonably
    svm_acc = metrics.accuracy_score(y_test, predicted_svm)
    rf_acc = metrics.accuracy_score(y_test, predicted_rf)

    assert svm_acc > 0.8, f"SVM accuracy too low: {svm_acc}"
    assert rf_acc > 0.8, f"Random Forest accuracy too low: {rf_acc}"