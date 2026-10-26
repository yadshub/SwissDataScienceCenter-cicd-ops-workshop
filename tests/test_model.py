from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from ci_cd.inference import predict
from ci_cd.train import load_data, train

# Min accuracy. Test fails if test accuracy after training dips below this.
MIN_ACCURACY = 0.8

# The model must beat the most-frequent-class baseline by at least this much.
MIN_BASELINE_MARGIN = 0.2


def check_beats_baseline(y_pred, y_test, baseline_pred):
    accuracy = accuracy_score(y_test, y_pred)
    baseline = accuracy_score(y_test, baseline_pred)
    assert accuracy >= baseline + MIN_BASELINE_MARGIN, (
        f"Accuracy {accuracy:.2f} does not beat baseline {baseline:.2f} "
        f"by {MIN_BASELINE_MARGIN:.2f}"
    )


def test_model_beats_baseline():
    train()

    X_train, X_test, y_train, y_test = load_data()
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    check_beats_baseline(predict(X_test), y_test, baseline.predict(X_test))


def test_baseline_fails_gate():
    X_train, X_test, y_train, y_test = load_data()
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    # Swapping the model for the baseline itself must trip the gate.
    try:
        check_beats_baseline(y_base, y_test, y_base)
    except AssertionError:
        return
    raise AssertionError("The baseline passed its own quality gate")


def test_model_accuracy():
    # Train and save the model, and load it back the way deployment would.
    train()

    _, X_test, _, y_test = load_data()
    y_pred = predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= MIN_ACCURACY, (
        f"Accuracy {accuracy:.2f} is below threshold {MIN_ACCURACY:.2f}"
    )
