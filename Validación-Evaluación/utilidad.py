#Utilidad para nuestros problemas del curso.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def ejemplo_regresion():
	from sklearn.datasets import make_regression
	from sklearn.linear_model import LinearRegression
	from adspy_shared_utilities import plot_two_class_knn

	X,y=make_regression(n_samples=300,n_features=1,n_informative=1,noise=30,random_state=0)
	X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

	plt.figure()
	plt.scatter(X,y,marker='o',s=50)
	lr=LinearRegression().fit(X_train,y_train)
	plt.plot(X,lr.coef_*X+lr.intercept_,'m-')
	plt.title('Ejemplo regresion lineal')


def ejemplo_clasificacion():
	from sklearn.datasets import make_classification
	from sklearn.neighbors import KNeighborsClassifier
	from adspy_shared_utilities import plot_two_class_knn

	X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=0.1,class_sep=0.5,random_state=0)
	X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

	plot_two_class_knn(X_train,y_train,3,'uniform',X_test,y_test)

def ejemplo_overfitting():
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures

	X=np.array([5,15,25,35,45,55]).reshape((-1, 1))
	y=np.array([28,11,3,8,29,30])

	plt.figure()
	#Lineal:
	plt.subplot(3,2,1)
	lr=LinearRegression().fit(X,y)
	plt.plot(X,y,'ro')
	plt.plot(X,lr.coef_*X+lr.intercept_,'k-')
	plt.title('Grado 1. R-squared={:.3f}'.format(lr.score(X,y)))

	#Grado 2:
	plt.subplot(3,2,2)
	poly=PolynomialFeatures(degree=2)
	X_poly=poly.fit_transform(X)
	lr=LinearRegression().fit(X_poly,y)
	plt.plot(X,y,'ro')
	plt.plot(X,lr.coef_[2]*X**2+lr.coef_[1]*X+lr.intercept_,'k-')
	plt.title('Grado 2. R-squared={:.3f}'.format(lr.score(X_poly,y)))

	#Grado 3:
	plt.subplot(3,2,5)
	poly=PolynomialFeatures(degree=3)
	X_poly=poly.fit_transform(X)
	lr=LinearRegression().fit(X_poly,y)
	plt.plot(X,y,'ro')
	plt.plot(X,lr.coef_[3]*X**3+lr.coef_[2]*X**2+lr.coef_[1]*X+lr.intercept_,'k-')
	plt.title('Grado 3. R-squared={:.3f}'.format(lr.score(X_poly,y)))

	#Grado 5:
	plt.subplot(3,2,6)
	poly=PolynomialFeatures(degree=5)
	X_poly=poly.fit_transform(X)
	lr=LinearRegression().fit(X_poly,y)
	plt.plot(X,y,'ro')
	plt.plot(X,lr.coef_[5]*X**5+lr.coef_[4]*X**4+lr.coef_[3]*X**3+lr.coef_[2]*X**2+lr.coef_[1]*X+lr.intercept_,'k-')
	plt.title('Grado 5. R-squared={:.3f}'.format(lr.score(X_poly,y)))

def ejemplo_logistica():
	from sklearn.linear_model import LogisticRegression

	X=np.array([0.5,0.7,1,1.8,2,3,3.5,4,4.5,5,5.2,5.5,6]).reshape((-1, 1))
	y=np.array([0,0,0,0,0,0,1,1,1,1,1,1,1])

	plt.subplot(2,1,1)
	plt.plot(X,y,'ro')
	plt.title('Curva para nuestro caso',x=0.3,y=0.7)

	X=[0,1,2,2.2,2.4,2.6,2.8,3,3.25,3.5,4,4.5,6]
	y=[0.01,0.05,0.05,0.1,0.2,0.3,0.4,0.5,0.64,0.8,0.85,0.9,0.95]

	plt.plot(X,y,'k-')

def ejemplo_claslin():
	X=np.arange(-2,2,0.1)
	y=np.arange(-2,2,0.1)

	X_1=[-2,-1.5,-1.4,-1,-1.7,-0.8,-0.6,-0.9,0,1.9]
	y_1=[0,0,1,0.2,0.3,1,2,2,0.3,2]

	X_2=[0,0.1,0.3,0.2,1,1.5,1.33,1.8,2,1.7]
	y_2=[-2,-1.9,-1.4,-0.6,0.4,1,0.1,1.7,1,1.3]

	plt.figure()

	plt.plot(X_1,y_1,'ro')
	plt.plot(X_2,y_2,'bo')
	plt.plot(X,y,'k-')
	plt.xlabel('Valor x')
	plt.ylabel('Valor y')
	plt.legend(['Clase 1','Clase 2','Recta de division'])
	plt.title('Clasificador Lineal')

def ejemplo_nolineal():
	X_1=[-6,-5,-4,0,1,2]
	y_1=[0,0,0,0,0,0]

	X_2=[-3,-2,-1]
	y_2=[0,0,0]

	plt.figure()

	plt.plot(X_1,y_1,'bo')
	plt.plot(X_2,y_2,'ro')
	plt.legend(['Clase 1','Clase 2'])
	plt.title(r'$v_i=(x_i)$')

def ejemplo_solnolineal():
	from sklearn.linear_model import LinearRegression

	X_1=[-6,-5,-4,0,1,2]
	y_1=[n**2 for n in X_1]

	X_2=[-3,-2,-1]
	y_2=[n**2 for n in X_2]


	xline=np.array([0,-4]).reshape((-1,1))
	yline=np.array([0,16])

	lr=LinearRegression().fit(xline,yline)
	X=np.arange(-6,2,1)

	plt.figure()

	plt.plot(X_1,y_1,'bo')
	plt.plot(X_2,y_2,'ro')
	plt.plot(X,X*lr.coef_+lr.intercept_,'k-')
	plt.legend(['Clase 1','Clase 2'])
	plt.title(r'$v_i=(x_i,x_i^2)$')

def ejemplo_mascomplejo():
	from sklearn.datasets import make_blobs
	from matplotlib.colors import ListedColormap

	cmap_bold=ListedColormap(['#FFFF00','#00FF00','#0000FF','#000000'])

	X,y=make_blobs(n_samples=100,n_features=2,centers=8,cluster_std=1.3,random_state=4)
	y=y%2

	plt.figure()
	plt.scatter(X[:,0],X[:,1],c=y,marker='.',s=50,cmap=cmap_bold)


def dibujar_knn():
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets

    n_neighbors=5

    #import some data to play with
    iris=datasets.load_iris()

    #we only take the first two features. We could avoid this ugly
    #slicing by using a two-dim dataset
    X=iris.data[:, :2]
    y=iris.target

    h=.02  #step size in the mesh
    
    #Create color maps
    cmap_light=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold=ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform','distance']:
        #we create an instance of Neighbours Classifier and fit the data.
        clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
        clf.fit(X, y)

        #Plot the decision boundary. For that, we will assign a color to each
        #point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
        y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
        xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])

        #Put the result into a color plot
        Z=Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

        #Plot also the training points
        plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
        plt.xlim(xx.min(),xx.max())
        plt.ylim(yy.min(),yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))

    plt.show()