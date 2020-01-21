import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
print(subsetClr)


#print(3**0.5)


#x = data['K_pct']
#y = data['Na_pct']
#col = data['lith_test'] # categorical variables for plotting need to be integers
#colormap = np.array(['r', 'g', 'b', 'r','g'])
#scatter = plt.scatter(x, y, c=colormap[col])
#plt.show()
