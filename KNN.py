## Tutorial on the implementation of K-NN on the MNIST dataset for the recognition of handwritten numbers.


from sklearn.datasets import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# DisplayImage method to display image data (optional method)
def displayImage(i):
    plt.imshow(digit['images'][i], cmap='Greys_r')
    plt.show()



#Display of dataset 
digit = load_digits() # loading the MNIST dataset
dig = pd.DataFrame(digit['data'][0:1700]) # Creation of a Panda dataframe
dig.head() # show the table below


# ## Let's display an image from the MNIST dataset!


displayImage(0) # display of the first image of the MNIST dataset
digit.keys()


train_x = digit.data
train_y =  digit.target


# ### Splitting of the MNIST data set into Training set and Testing Set. With:
# * 75% in Training set
# * 25% in Testing set


x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.25)


# ## Instantiation and training of a K-NN classifier with K = 7


KNN = KNeighborsClassifier(7)
KNN.fit(x_train, y_train)


# ## Calculation of the performance scoring of our 7-NN model


# Accuracy compared to test data
print(KNN.score(x_test,y_test))


# ## Prediction test of our model on a figure not yet seen


# Display an element of the image format matrix
test = np.array(digit['data'][1726])
test1 = test.reshape(1,-1)
displayImage(1726)
#Predicition Result : 3

#test = np.array(digit['data'][1720])
#test1 = test.reshape(1,-1)
#displayImage(1720)
#Predicition Result : 8

# Print Prediction
print("Predicted Number:", KNN.predict(test1))
