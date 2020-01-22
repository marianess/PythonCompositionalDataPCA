import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# read in data
data = pd.read_csv("GV003.csv") 

# extract column names as pandas.index
colnames  = data.columns[5:66] # might need to extract only geochemical variable names

# convert column names to list
colnames = colnames.tolist()

# subset data by selected columns
datasubset = pd.DataFrame(data[colnames])

# create an empty dataframe to store the centered-logratio(-clr)-transformed values 
subsetClr = pd.DataFrame()

# loop over rows of the dataframe
for index, row in datasubset.iterrows():
    # calculate the product of each row as a pre-step for geometric mean calculations
    ProductByRow = np.prod(row) # an array of storing the product of each row
        # calculate geometric mean of each row
    GeometricMean = ProductByRow**(1/(len(row))) # geometric mean of a composition, i.e. each row
    # create an empty list to store the clr values of each row
    NewRow = []
    # loop over every element of a row
    for RowIndex in range(len(row)):
        # calculate a clr score of each element of a row
        ElementClr = math.log(row[RowIndex] / GeometricMean) # generating items in a row
        # append each clr score into a new list
        NewRow.append(ElementClr)
    # create a multiple 61X1 dataframes from individual 'NewRow' lists
    NewRow = pd.DataFrame(NewRow)
    # transpose multiple dataframes into 1x61 row-based dataframes
    NewRow = NewRow.T
    # concatenate row-based dataframes into a single subsetClr dataframe
    subsetClr = pd.concat([subsetClr, NewRow], axis=0)
# rename the header line according to original column names
subsetClr.columns = colnames

# creating a covariance matrix of SubsetClr
#X_std = subsetClr.to_numpy()
X_std = StandardScaler().fit_transform(subsetClr) ## NEEDS TO BECOMPOSITIONALLY CENTERED HERE!!!!
print(type(X_std))

# create a covariance matrix of clr transformed data
covmat  = subsetClr.cov()

# perform extraction of eigenvectors and eigenvalues from the covariance matriz
eig_vals, eig_vecs = np.linalg.eig(covmat)

# make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# calculate the proportion of explained variance and cumulative explained variane
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# define number of PCs to visualize on the elbow plot
PCnumber = 5 # can be user defined

# create an elbow plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(PCnumber), var_exp[0:PCnumber], alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(PCnumber), cum_var_exp[0:PCnumber], where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()

# create a projection matrix W which contains loadings
matrix_w = np.hstack((eig_pairs[0][1].reshape(len(covmat),1),
                      eig_pairs[1][1].reshape(len(covmat),1)))
Y = X_std.dot(matrix_w) # Y = U X W^transposed
# Y is a matrix that contains scores
# U is the original matrix of data, i.e. X_std
# W is the matrix of loadings 

# get the values of lithology/cluster AT THE MOMENT THE COLUMNS IS CREATED BY MARIA TO TEST
y = data['lith_test'].values 

# get the uniqe values of the lithology/cluster
yUnique = list(set(y)) 

# set up a palette of colours, up to 10 at the moment
palette = ['blue', 'red', 'green','yellow','cyan','violet','grey','saddlebrown','pink','orange']

# create a plot of scores
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((yUnique), # lab is label, col is color
                        (palette[0:len(yUnique)])):
                        plt.scatter(Y[y==lab, 0],
                                    Y[y==lab, 1],
                                    label=lab,
                                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

# create a plot of loadings
# obtain loadings onto PC1
xvector = matrix_w.T[0] 

# obtain loadings onto PC2
yvector = matrix_w.T[1]

for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i], yvector[i],
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i], yvector[i],
             list(colnames)[i], color='r')
    plt.xlim(min(xvector)-0.05,max(xvector)+0.05)
    plt.ylim(min(yvector)-0.05,max(yvector)+0.05)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

plt.show()

