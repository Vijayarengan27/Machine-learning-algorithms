import numpy as np
import math




class MultiGaussClassify:
    def __init__(self,k,d,diag):
        self.k = k
        self.d = d
        self.diag = diag
        
        self.mean = 0
        self.covariance= np.identity(d)
        self.p_c =[] 
               
    def fit(self,X,y):
        mean = []
        covariance = []
        l = X.data.shape[1]
        n = X.data.shape[0]
        ind,frequency = np.unique(y,return_counts = True)
        self.p_c = frequency/l 
        
        for i in range(0,self.k):
            indices = []
            for index,target in enumerate(X):
                if y[index] == i:
                    indices.append(index)
            class_variables = np.asmatrix(X[indices])
            mean_class = np.mean(class_variables,axis = 0)
            Cv = np.subtract(class_variables,np.matmul(np.ones((class_variables.shape[0],1),dtype = int),mean_class))
            Covariance = np.divide(np.matmul(np.transpose(Cv),Cv),class_variables.shape[0])
            mean.append(mean_class)
            
            if self.diag:
                Covariance = np.diag(np.diag(Covariance)) 
            if np.linalg.matrix_rank(Covariance) < l:
                Covariance = np.add(Covariance,np.identity(l))
            
            covariance.append(Covariance)
    
              
            self.mean = mean
            self.covariance = covariance
            
            
        
        return mean,covariance
    
    def predict(self,X):
        result = []
        n = X.data.shape[0] 
        for j in range(0,n):
            compare = []
            for i in range(0,self.k):
                
                determinant = np.linalg.det(self.covariance[i])
                
                constant = 0.5 * math.log(determinant)
                X_c = np.subtract(X[j],self.mean[i])
                inside_expo = 0.5 * np.matmul(X_c,np.matmul(np.linalg.inv(self.covariance[i]),X_c.transpose()))
                inside_expo = np.asscalar(inside_expo)
                
                g = math.log(self.p_c[i])-constant - inside_expo
                
                compare.append(g)
            clas = np.argmax(g)
            result.append(clas)
            
        
        return result
        
