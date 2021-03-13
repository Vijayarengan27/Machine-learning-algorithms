import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))
sigmoid_vector = np.vectorize(sigmoid)           #to calculate the sigmoid function 

class MyLogisticreg2:
    def __init__(self,d):
        self.w = 0.02*np.random.random_sample((d+1,)) - 0.01 # we can enter any initial w values here, I have entered these as an example. The entered values should be close to a minima
        
    
    def fit(self,X,y):
        self.w = self.weights(X,y)
        return self


    def predict(self,X):
        ypred = self.predictlabels(X)
        return ypred


    def weights(self,X,y):
    	n = X.shape[0]
    	mean = np.mean(X,axis = 0)
    	sd = np.std(X,axis = 0)                        
    	X_normalized = (X-mean)/sd         #normalizing the feature values
    	X_1 = np.ones((n,))
    	X_new = np.insert(X_normalized, 0, X_1, axis=1)    #adding a column of ones for w0  
    	updatedWeights = self.convergence(X_new, y)
    	return updatedWeights


    def predictlabels(self,X_test):
    	n = X_test.shape[0]
    	mean = np.mean(X_test,axis = 0)
    	sd = np.std(X_test,axis = 0)                        
    	X_test_normalized = (X_test-mean)/sd
    	X_test_1 = np.ones((n,))
    	X_test_new = np.insert(X_test_normalized, 0, X_test_1, axis=1) #inserting column of ones to get the w0's 
    	ypred = sigmoid_vector(np.matmul(X_test_new,self.w))
    	ypred = ypred > 0.5
    	return ypred


    def convergence(self,X_train,y_train):
    	w_prev = self.w
    	w_new = w_prev
    	n = X_train.data.shape[0]
    	accuracy = 0.01
    	learning_rate = 0.001
    	max_iterations = 10000
    	n_iterations = 0
    	while n_iterations < max_iterations:
        	h_prev = sigmoid_vector(np.matmul(X_train,w_prev))
        	loss_prev = -np.sum(np.multiply(y_train, np.log(h_prev)) + np.multiply((1-y_train), np.log(1-h_prev)))
        	gradient = -np.matmul(X_train.T, (y_train-h_prev))
        	delta = - (learning_rate*gradient)
        	w_new = w_prev + delta                            # updating w_new using the delta calculated
        	h_new = sigmoid_vector(np.matmul(X_train, w_new))
        	loss_new = -np.sum(np.multiply(y_train, np.log(h_new)) + np.multiply((1-y_train), np.log(1-h_new)))
        	conv_value = abs(loss_new-loss_prev)              # calculating absolute difference between two successive losses for convergence check
        	
        	# checking the condition for convergence
        	if conv_value < accuracy:
            		break
        	w_prev = np.copy(w_new)
        	n_iterations += 1
        
    	return w_new
    
