import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Kmeans: 
    def __init__(self,data,X_cols,n_cluster,random_state):
        self.X = data.loc[:,X_cols].values        
        self.kmeans = KMeans(n_clusters=n_cluster,
                        random_state=random_state)
        
    def train(self):
        fit = self.kmeans.fit(self.X)
        return(fit)
        
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X_test= test_data.loc[:,X_cols].values
        predict = self.kmeans.predict(X_test)
        return(predict)
    
    
class LogRegression:
    def __init__(self,data,X_cols,y_col,random_state,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,
                                                    test_size = test_size)   
        self.logr = LogisticRegression(random_state=random_state)
                  
    def train(self):
        fit = self.logr.fit(self.X_train,self.y_train)
        train_score = self.logr.score(self.X_train,self.y_train)
        val_score = self.logr.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.logr.predict(X)
        return(predict)
    
        
class SGD:
    def __init__(self,data,X_cols,y_col,max_iter,tol,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,
                                                    test_size = test_size)
        self.SGD = SGDClassifier(max_iter=max_iter, tol=tol)
                  
    def train(self):
        fit = self.SGD.fit(self.X_train,self.y_train)
        train_score = self.SGD.score(self.X_train,self.y_train)
        val_score = self.SGD.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.SGD.predict(X)
        return(predict)
    
    
class BayesianReg():
    def __init__(self,data,X_cols,y_col,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                    test_size = test_size)
        self.bayesr = BayesianRidge()
                  
    def train(self):
        fit = self.bayesr.fit(self.X_train,self.y_train)
        train_score = self.bayesr.score(self.X_train,self.y_train)
        val_score = self.bayesr.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.bayesr.predict(X)
        return(predict)
    
    
class RandomForest():
    def __init__(self,data,X_cols,y_col,max_depth,random_state,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,
                                                    test_size = test_size)
        self.randf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
                  
    def train(self):
        fit = self.randf.fit(self.X_train,self.y_train)
        train_score = self.randf.score(self.X_train,self.y_train)
        val_score = self.randf.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.randf.predict(X)
        return(predict)

class KNN():
    def __init__(self,data,X_cols,y_col,n_neighbors,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,
                                                    test_size = test_size)
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
                  
    def train(self):
        fit = self.neigh.fit(self.X_train,self.y_train)
        train_score = self.neigh.score(self.X_train,self.y_train)
        val_score = self.neigh.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.neigh.predict(X)
        return(predict)

class MLP():
    def __init__(self,data,X_cols,y_col,
                 random_state,epochs,activation,solver,learning_rate,
                 batch_size,test_size,standardize):
        X = data.loc[:,X_cols].values
        y = data.loc[:,y_col].values
        if standardize == True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,
                                                    test_size = test_size)
        self.nn = MLPClassifier(random_state=random_state, max_iter=epochs, activation=activation,
                                solver=solver,learning_rate_init=learning_rate, batch_size=batch_size)
                  
    def train(self):
        fit = self.nn.fit(self.X_train,self.y_train)
        train_score = self.nn.score(self.X_train,self.y_train)
        val_score = self.nn.score(self.X_test,self.y_test)
        return("Training Score: ", train_score, "Validation Score: ", val_score)
    
    def inference(self,test_data,X_cols):
        test_data = pd.read_csv(test_data)
        X = test_data.loc[:,X_cols].values
        predict = self.nn.predict(X)
        return(predict)


#Unquote to test the models uploading data and parameters on python.
"""""
print("Kmeans results:")
kmeans = Kmeans("iris_data.csv",['SepalLengthCm','SepalWidthCm',
                                               'PetalLengthCm','PetalWidthCm'], 3,0)
print(kmeans.train(),kmeans.inference("iris_data.csv",['SepalLengthCm','SepalWidthCm',
                                               'PetalLengthCm','PetalWidthCm']))

print("LogRegression results:")
logr = LogRegression("iris_data.csv",['SepalLengthCm','SepalWidthCm',
                                               'PetalLengthCm','PetalWidthCm'],'Species',0, 0.33,True)
print(logr.train())

print("SGD results:")
SGD = SGD("iris_data.csv",['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],'Species',1000,None,0.33,True) 
print(SGD.train())

print("Bayesian Regression results:")
bayesreg = BayesianReg("iris_data.csv",['SepalLengthCm','SepalWidthCm','PetalLengthCm'],"PetalWidthCm", 0.33, True)
print(bayesreg.train())

print("Random Forest results:")
randomforest = RandomForest("iris_data.csv",['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],'Species',2,0,0.33,True)
print(randomforest.train())

print("Knn results:")
knn = KNN("iris_data.csv",['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],'Species',8, 0.33,True)
print(knn.train())

print("Multi-Layer Perceptron results:")
mlp = MLP("iris_data.csv",['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],'Species',1,200,"relu",'adam',0.001,
                 'auto',0.33,True)
print(mlp.train())
"""""