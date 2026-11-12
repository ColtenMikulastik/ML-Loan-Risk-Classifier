from split import split_data
from sklearn.tree import DecisionTreeClassifier


def main():
    """ trains tree from given financial data """
    (X_train, X_test, y_train, y_test) = split_data()

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train) 

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()

