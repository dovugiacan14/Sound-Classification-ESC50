import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

def split_and_normalize(dataset, features, seed): 
    train = dataset.sample(frac=0.8, random_state=seed)
    test = dataset.drop(train.index)

    X_train = train[features]
    X_test = test[features]
    y_train = train['target']
    y_test = test['target']

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test
