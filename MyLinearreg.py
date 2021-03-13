import numpy as np

class MyLinearreg:
	
	def __init__(self,d):
		self.w = _____________ #Here we have to enter the initial w values. They should be near a minima.
	


	def fit(self,X,y):
 		X_final = preprocess(X)
		self.w = self.gradientdescent(X_final,y)
		return w

	def predict(self,X):
 		X_final= preprocess(X)
		y_pred = np.matmul(X,self.w)
		return y_pred

	def preprocess(self,X):
  		n = X.shape[0]
 	 	mean = np.mean(X,axis = 0)
  		sd = np.std(X,axis=0)
  		X_normalized = (X - mean) / sd  #normalizing the numpy matrix
  		one = np.ones((n,))
  		X_final = np.insert(X_normalized,0,one,axis = 1) #adding a column of ones in the matrix so as to accomodate the constant term in the linear regression equation
  		return X_final 

	def gradientdescent(self,X,y):  # the convergence value,accuracy,step size and iterations can be changed according to preferences
  		w_prev = self.w
 		w_new = w_prev
  		n = X.shape[0]
  		accuracy = 0.01 #convergence check value
  		learning_rate = 0.001 #step size or learning rate of the descent
  		max_iterations = 10000 # maximum iterations of the gradient descent method
  		n_iterations = 0
  		while n_iterations < max_iterations:
    			delta = 2 * np.matmul(X.T, (np.matmul(X,w_prev) - y))
    			w_new = w_prev - learning_rate * delta
    			l_prev = np.matmul(X,w_prev) - y
    			loss_prev = np.linalg.norm(l_prev) ** 2
    			l_new = np.matmul(X,w_new)
    			loss_new = np.linalg.norm(l_new) ** 2
    			convergence_value = abs(loss_new - loss_prev)  #finding whether the loss change is significant to convergence value
    			if convergence_value < accuracy:
      				break
    			w_prev = np.copy(w_new)
    			n_iterations += 1
  		return w_new