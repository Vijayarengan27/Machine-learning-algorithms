import numpy as np



def k_fold_method(data,k):
    len = data.shape[0]
    indices = list(range(len))
    train_indices = []
    test_indices = []
    for i in  range(k):
        #arranging the in the form of indices from the given range
        test = list(np.arange((i*(int(len/k))),((i+1)*(int(len/k)))))
        test_indices.append(test)
        #getting train indices by taking rest of the indices from test indices
        train = list(set(indices) ^ set(test))
        train_indices.append(train)
    return train_indices,test_indices



def my_cross_val(method,X,y,k):
    error_rates = []
    train,test = k_fold_method(X,k)
    for i in range(0,k):
       X_train,y_train = X[train[i]],y[train[i]]
       X_test,y_test = X[test[i]],y[test[i]]
       method.fit(X_train,y_train)
       pred = method.predict(X_test)
       #making the matrix of 1s and 0s according to the inequality
       errors = (y_test != pred).astype(int)
       error_rates.append(np.sum(errors)/(y_test.data.shape[0]))
       print(f'The error rates for the fold {i}')
       
    error_mean = np.sum(error_rates)/k
    error_sd = np.std(error_rates)
    print(error_rates)
    print(error_mean,error_sd)
    return error_rates,error_mean,error_sd


