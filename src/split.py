
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data_file:str="data/german_encoded.csv", train_size:float=.8, random_state:int=42, stratify:bool=True):
    """
    loads and splits data from file
    returns:
        X_train, X_test, y_train, y_test
    """

    df = pd.read_csv(data_file)

    X = df.drop(columns=["bad_credit"])
    y = df["bad_credit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,      # 80% for training data
        test_size=(1-train_size),   # 20% for test and validation
        random_state=random_state,  # set to int for reproduceability 
        stratify=y if stratify else None,         # 30% split of "bad credit" class in training and test
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}") # print (num_rows, num_feature) for train and test
    print(f"Train label balance:\n{y_train.value_counts(normalize=True)}") # % of training labeled by class
    print(f"Test label balance:\n{y_test.value_counts(normalize=True)}") # same as above for test (shoudl be same)

    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    split_data()