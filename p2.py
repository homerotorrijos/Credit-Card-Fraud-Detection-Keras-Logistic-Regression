import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def carga_csv(file_name):
	"""carga el fichero csv especificado y los devuelve en un array de numpy"""
	datos = read_csv(file_name,header=None).values
	datos = datos.astype(float)
	return datos

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def h(theta,X):
	return np.dot(X,theta)

def prob(theta,X):
	return sigmoid(h(theta,X))

def coste(theta,X,Y):
	m = X.shape[0]
	coste_total = -(1/m) * np.sum(Y * np.log(prob(theta,X)) + (1 - Y) * np.log(1-prob(theta,X)))
	return coste_total

def gradiente(theta,X,Y):
	m = X.shape[0]
	return (1/m) * np.dot(X.T,prob(theta,X) - Y)

def fit(theta,X,Y):
	result = opt.fmin_tnc(func=coste,x0=theta,fprime=gradiente,args=(X,Y))
	return result[0]


def evaluate(theta,X,Y):
	m = X.shape[0]
	result = prob(theta,X)
	values = np.where(result >= 0.5,1.,0.)
	return np.sum(values == Y)/m

def regresionLogistica(X,y):
	initial_theta = np.zeros((X.shape[1], 1))
	theta = fit(initial_theta,X,y)
	return theta
	
def ejecutar():
	path = 'creditcard.csv'
	datos = carga_csv(path)
	X = datos[:,1:-2]
	Y = datos[:,-1]
	m = np.shape(X)[0]
	n = np.shape(X)[1]
	X_ = np.hstack([np.ones([m,1]),X])

	initial_theta = np.array([0,0,0])
	theta = fit(initial_theta,X_,Y)
	
	porcentage = evaluate(theta,X_,Y)
	print(porcentage)
	pinta_frontera_recta(X,Y,theta)

def main():
	ejecutar()

#main()