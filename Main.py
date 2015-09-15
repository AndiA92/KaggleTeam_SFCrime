from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
import numpy as np
import collections
import csv

trainFile = '../datasets/train.csv'
testFile = '../datasets/test.csv'
resultsFile = '../datasets/results.csv'

# Read train set file
trainSetFromFile = genfromtxt(trainFile, delimiter=',', dtype='string')

# Get unique numerical indexes for distinct days, categories, districts
# These are alphabetically sorted. i.e : [Monday, Thursday, Friday, Monday] will lead to [1,2,0,1]
daySet_Train = np.unique(trainSetFromFile[1:, 3], True, True)
categorySet_Train = np.unique(trainSetFromFile[1:, 1], True, True)
districtSet_Train = np.unique(trainSetFromFile[1:, 4], True, True)

# The first element of the array gives the distinct day/category/district values
daySet_value = daySet_Train[0]
categorySet_value = categorySet_Train[0]
districtSet_value = districtSet_Train[0]

# The second element of the array gives the index of the first appearance of each value
daySet_firstAppearance = daySet_Train[1]
categorySet_firstAppearance = categorySet_Train[1]
districtSet_firstAppearance = districtSet_Train[1]

# The third element of the array gives the unique indexes for values in the first array
daySet_key = daySet_Train[2]
categorySet_key = categorySet_Train[2]
districtSet_key = districtSet_Train[2]

# Creates pairs of day-district values
trainSet = np.column_stack((daySet_key, districtSet_key))

# Performs train operation with Gaussian Naive Based
gnb = GaussianNB()
y_pred = gnb.fit(trainSet, categorySet_key)

# Reads test file
testSetFromFile = genfromtxt(testFile, delimiter=',', dtype='string')

# Get unique indexes for days and districts from test file
daySet_Test = np.unique(testSetFromFile[1:, 2], True, True)[2]
districtSet_Test = np.unique(testSetFromFile[1:, 3], True, True)[2]

# Creates pairs of day-district values
testSet = np.column_stack((daySet_Test, districtSet_Test))

output = {}

for i in range(0, len(testSet)):
    # Gets probability array for day-district pairs, length = number of distinct categories from train set
    probabilities = y_pred.predict(testSet[i])
    # Fill the output dictionary values
    # Key: category, values : list of probabilities for each day-district pair
    for j in range(0, len(probabilities[0])):
        currentCategory = categorySet_value[j]
        currentValue = output.get(currentCategory, None)

        if currentValue is not None:
            output[currentCategory] = np.append([currentValue], (probabilities[0])[j])
        else:
            output[currentCategory] = (probabilities[0])[j]

# Alphabetically sorts the dictionary
sorted_categories = collections.OrderedDict(sorted(output.items()))

# Writes results to output file
with open(resultsFile, 'w') as csvfile:
    fieldnames = sorted_categories.keys()
    fieldnames.insert(0, "ID")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for i in range(0, len(testSet)):

        toWrite = collections.OrderedDict({})
        for key in fieldnames:
            if key == "ID":
                toWrite["ID"] = i
            else:
                toWrite[key] = (sorted_categories[key])[i]
        writer.writerow(toWrite)
