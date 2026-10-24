"""Handwritten Digit Recognition - recognises 0-9 digits from 8x8 images using an SVM."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )

    search = GridSearchCV(SVC(), {"C": [1, 10], "gamma": [0.001, 0.0005]}, cv=3)
    search.fit(X_train, y_train)
    print(f"Best settings: {search.best_params_}")

    model = search.best_estimator_
    pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}\n")
    print(classification_report(y_test, pred))

    # Show the first 10 test images with their predictions
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for ax, image, label, guess in zip(axes.ravel(), X_test, y_test, pred):
        ax.imshow(image.reshape(8, 8), cmap="gray_r")
        ax.set_title(f"pred {guess}", color="green" if guess == label else "red")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("predictions.png")
    print("Saved predictions.png")


if __name__ == "__main__":
    main()
