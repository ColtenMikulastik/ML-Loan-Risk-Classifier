from split import split_data
from sklearn.linear_model import LogisticRegression


def main():
    """ trains a logistic regression model on the given financial data"""
    (X_train, X_test, y_train, y_test) = split_data()

    model = LogisticRegression(max_iter=1000) # setting maximum iterations for training?
    model.fit(X_train, y_train) # train model

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()