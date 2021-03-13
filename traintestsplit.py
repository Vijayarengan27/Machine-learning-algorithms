import numpy as np

def split(X,y,pi):
 random_shuffle = []
 train = []
 test = []
 len = X.data.shape[0]
 ratio = pi*len
 #getting the list of all indices
 index = list(np.arange(len))
 #shuffling the indices
 np.random.shuffle(index)
 random_shuffle = index
 #getting the pi ratio of shuffled indices as train indices
 train = random_shuffle[int(ratio):]
 #getting the rest of the shuffled indices as test indices
 test = random_shuffle[:int(ratio)]
 X_train,y_train,X_test,y_test = X[train],y[train],X[test],y[test]
 return X_train,y_train,X_test,y_test



def my_train_test(method,X,y,pi,k):
 error_rates = []
 for i in range(k):
     X_train,y_train,X_test,y_test = split(X,y,pi)
     method.fit(X_train,y_train)
     pred = method.predict(X_test)
     #getting a matrix of 1s and 0s according to the inequality
     errors = (y_test != pred).astype(int)
     error_rates.append(np.sum(errors)/(y_test.data.shape[0]))
 mean = np.sum(error_rates)/k
 standard_deviation = np.std(error_rates) 
 return error_rates,mean,standard_deviation




