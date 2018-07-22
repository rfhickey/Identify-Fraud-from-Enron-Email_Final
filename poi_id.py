#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import defaultdict

### Task 1: Select what features you'll use.
### used_features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
original_features_list = ['poi','salary','total_payments','bonus','total_stock_value',
                 'exercised_stock_options','long_term_incentive','to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi','shared_receipt_with_poi'] 

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# I want to get a sense of the data I am working with here
# Number of People in the Dataset: 146

number_of_people = len(data_dict)    
    
print number_of_people   

# Of These 146 People, I Wanted to See How Many are Persons of Interest. There
# are 18
counter_poi = 0
for person, feature in data_dict.iteritems():
    if feature['poi']:
        counter_poi += 1

print counter_poi

# We Were Given a List of Names of People from a USA Today Article Who Were Seen
# As Persons of Interest. Do These Match Up with the POIs in the
# final_project_dataset.pkl file? The Answer Here Is No As There are 35 In That
# List

with open("poi_names.txt") as f:
    pois_total_in_list = len(f.readlines()[2:])
print(pois_total_in_list)
  
# Let's See How Many Features There Are for Each Person. To do this, I Printed
# The Length of the Dictionary Storing the Values for the first person in the 
# PDF Provided in the final_project folder. There are 21 features, which 
# Matches with the Number Provided in the Udacity Final Project Page 
print (len(data_dict['ALLEN PHILLIP K']))

### Task 2: Remove outliers

# Let's Do a Quick Test to See What Some of the Most Intuitively Easy to
# Understand Featues Look Like

features_outlier_test_1 = ['salary', 'total_stock_value']

features = featureFormat(data_dict, features_outlier_test_1, sort_keys = True) 

for i in features:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value)
   

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.show()

# I remember this from the Udacity lessons that we had a "TOTAL" figure in here.
# Let's make sure we remove that and anything else we find when we do a manual
# scan of the data.

# I wanted to see if there were any odd keys in the dataset that needed to 
# be removed. I found "TOTAL" and something called "THE TRAVEL AGENCY IN THE
# PARK" (that has NaNs for all entries) which are not individuals like 
# the other entries, so I removed them
for key, value in data_dict.iteritems():
    print key

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

# I'm interested in seeing how many of the features in the data are missing 
# values and how many values they are missing

missing_data = defaultdict(float)
value_list = next(data_dict.itervalues()).keys()
for key in data_dict.itervalues():
    for value in value_list:
        if key[value] == "NaN":
            missing_data[value] += 1
            
print missing_data

# I also want to see what percentage of each feature is missing, so I have created
# the code below based on the cleaned data with 144 people with emails

missing_data_1 = defaultdict(float)

for key in missing_data.itervalues():
    for value in missing_data: 
        missing_data_1[value] = round(missing_data[value] / 144, 2)

print missing_data_1

# You can see from these results that there is a lot of missing data here. I would
# say that we would not want to include >55% NaNs, so remove  deferral_payments, restricted_stock_deferred,
# deferred_income, loan_advances, director_fees, and long_term_incentive as
# features when training our machine learning algorithm. Here is the new data set
# with long_term_incentive removed

features_list_v2 = ['poi','salary','total_payments','bonus','total_stock_value',
                 'exercised_stock_options','to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi','shared_receipt_with_poi']   

# Let's also get a bit more aggressive, exhaustive, and experimental with the data cleaning
# for features that have missing data than the exclusion of features with >55% NaNs 
# (or missing data) threshold we used to construct features_list_v2 above. 
# In this features list, features_list_v3, we are are exluding features that 
# have more than >=40 % missing data. We thus remove 'to_messages', 'bonus', 'shared_receipt_with_poi', 
# 'from_poi_to_this_person', 'from_messages', and 'from_this_person_to_poi'

features_list_v3 = ['poi','salary','total_payments','total_stock_value',
                 'exercised_stock_options'] 

# Finally, let's also get very aggressive, exhaustive, and experimentak with the data cleaning
# for features that have missing data than the exclusion of features with >40% NaNs 
# (or missing data) threshold we used to construct features_list_v2 above. 
# In this features list, features_list_v4, we are are exluding features that 
# have more than >=20 % missing data. We thus remove 'salary' and 'exercised_stock_options'
# leaving us with two features. 

features_list_v4 = ['poi','total_payments','total_stock_value'] 

# Since There Seem to Be Decently Robust Data for salary and total_stock_value,
# let's visualize those data again after we have cleaned up the data. 

features_outlier_test_1 = ['salary', 'total_stock_value']

features = featureFormat(data_dict, features_outlier_test_1, remove_any_zeroes=True) 

for i in features:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value)

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.xlim(0, 1200000)
plt.ylim(0, 50000000)
plt.show()

# And let's add a regression line to the data to get a sense of it. I used this
# as a reference: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

salary_feature = ['salary']
total_stock_value_feature = ['total_stock_value']

salary_train = featureFormat(data_dict, salary_feature, remove_any_zeroes=True)
total_stock_value_train = featureFormat(data_dict, total_stock_value_feature, 
                                        remove_any_zeroes=True)


print len(salary_train)
print len(total_stock_value_train)

# I found there were 94 salary_train data points, so I limited myself the
# dataset to these points so the code didn't throw an error.

salary_train = salary_train[:94]
total_stock_value_train = total_stock_value_train[:94]


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(salary_train, total_stock_value_train)

pred_total_stock_value = reg.predict(salary_train)

features_outlier_test_1 = ['salary', 'total_stock_value']

# I'm reusing the code from above to plot the points here again under the
# linear regression
features1 = featureFormat(data_dict, features_outlier_test_1, remove_any_zeroes=True) 

for i in features1:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value, color = 'blue')

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.xlim(0, 1200000)
plt.ylim(0, 50000000)
plt.plot(salary_train, pred_total_stock_value, color="r")
plt.show()

print(reg.coef_)
print(reg.intercept_)

print(reg.score(salary_train, pred_total_stock_value))

# Looking at this linear regression we can see some data points that fall far
# outside the data as predicted by the linear regression. Despite this, we do
# not want to eliminate these points since they could reveal interesting POI.
# As you would expect, the r^2 value is 1 because we are using the same 
# training data for the fit as we are using for the prediction of total_stock_value,
# so the predictions should fall perfectly on the regression. I just wanted to
# model the data here using linear regression in Python.   
 


### Task 3: Create new feature(s)
# One feature I think it will be important to add is something that helps us 
# see when total_stock_value and salary are out of alignment. For example, if an 
# employee received a huge amount of stock but a small salary, something might be up.

features_list = ['poi','salary','total_payments','bonus','total_stock_value',
                 'exercised_stock_options','to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi','shared_receipt_with_poi'] 


for key, value in data_dict.iteritems():
    if value["salary"]!='NaN' and value["total_stock_value"]!='NaN':
        data_dict[key]["stock_salary_ratio"] = data_dict[key]["total_stock_value"]/data_dict[key]["salary"]
    else: 
        data_dict[key]["stock_salary_ratio"] = 'NaN'
    
# print data_dict

features_list.append("stock_salary_ratio")

print features_list
     
# Let me also look at KBest to see which features to keep or drop here. I used 
# this as a reference: https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work

from sklearn.feature_selection import SelectKBest

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k='all')
k_best.fit(features, labels)
scores = k_best.scores_
pairs = zip(features_list[1:15], scores)

print pairs

# We can see from this that there are six features that have a score of over
# 8 when using SelectKBest. These are "total_payments", "shared_recepit_with_poi", 
# "salary", "bonus", "total_stock value",
# and "exercized stock options". Let's remove everything except for these so
# we do no overfit out data

features_list.remove("to_messages")
features_list.remove("from_poi_to_this_person")
features_list.remove("from_messages")
features_list.remove("from_this_person_to_poi")
features_list.remove("stock_salary_ratio")

print features_list

# But we also want to exhastively test whether 8 is the best breaking point
# for a k value to exclude everything below. So I have also tested the 
# algorithm including features with a k of greater than 18 and a k of greater than 2

# K of greater than 18
#features_list.remove("total_payments")
#features_list.remove("to_messages")
#features_list.remove("from_poi_to_this_person")
#features_list.remove("from_messages")
#features_list.remove("from_this_person_to_poi")
#features_list.remove("shared_receipt_with_poi")
#features_list.remove("stock_salary_ratio")
#
#features_list_v5 = features_list
#
#print features_list_v5

# K of greater than 2

#features_list.remove("to_messages")
#features_list.remove("from_messages")
#features_list.remove("stock_salary_ratio")
#
#features_list_v6 = features_list
#
#print features_list_v6

### Store to my_dataset for easy export below.
my_dataset = data_dict

# This was the features list is tried that excluded >=40% missing data

# features_list = features_list_v3

# This was the features list is tried that excluded >20% missing data

# features_list = features_list_v4

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print len(labels)
print len(features)

# Let's also split the data into training and test datasets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Developed Naive Bayes Classifier and Tested Accuracy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf1 = GaussianNB()

# CV_SSS_NB = StratifiedShuffleSplit(labels, 20)

# clf1 = grid_search.GridSearchCV(clf_nb, cv=CV_SSS_NB, scoring = "f1")

clf1.fit(features_train, labels_train)

pred = clf1.predict(features_test)

accuracy1 = accuracy_score(pred, labels_test)

print 'Precision_NB:', precision_score(labels_test, pred)
print 'Recall_NB:', recall_score(labels_test, pred)
print 'Accuracy_NB:', accuracy1

# Developed Support Vector Machines Classifier and Tested Accuracy
from sklearn.svm import SVC

clf2 = SVC()

clf2.fit(features_train, labels_train)

pred = clf2.predict(features_test)

accuracy2 = accuracy_score(pred, labels_test)

print 'Precision_SVM:', precision_score(labels_test, pred)
print 'Recall_SVM:', recall_score(labels_test, pred)
print 'Accuracy_SVM:', accuracy2

# Developed Support Vector Machines Classifier and Tested Accuracy
from sklearn import tree

clf3 = tree.DecisionTreeClassifier()
clf3.fit(features_train, labels_train)

pred = clf3.predict(features_test)

accuracy3 = accuracy_score(pred, labels_test)

print 'Precision_DT:', precision_score(labels_test, pred)
print 'Recall_DT:', recall_score(labels_test, pred)
print 'Accuracy_DT:', accuracy3


# Developed K-Nearest Neighbors Classifier and Tested Accuracy
from sklearn.neighbors import KNeighborsClassifier

clf4 = KNeighborsClassifier()

clf4.fit(features_train, labels_train) 

pred = clf4.predict(features_test)

accuracy4 = accuracy_score(pred, labels_test)

print 'Precision_KNC:', precision_score(labels_test, pred)
print 'Recall_KNC:', recall_score(labels_test, pred)
print 'Accuracy_KNC:', accuracy4


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# I am going to use the K-Nearest Neighbors since it is
# the one I can most get my head around conceptually (maybe not the best
# reason to select it) and had the greatest accuracy above. 

from sklearn.neighbors import KNeighborsClassifier

parameters1 = {'n_neighbors': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45], 
              'weights': ('distance', 'uniform')}
knc = KNeighborsClassifier()
clf5 = grid_search.GridSearchCV(knc, parameters1)
clf5.fit(features, labels)

clf5.best_params_

# Of the parameters I have chosen, it looks like the best one is an 
# n-neighbors of 10 and a weight of "distance."

from sklearn.neighbors import KNeighborsClassifier

clf6 = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')

# clf6 = KNeighborsClassifier()

clf6.fit(features_train, labels_train) 

pred = clf6.predict(features_test)

accuracy = accuracy_score(pred, labels_test)

print accuracy

# Ok, the accuracy went down slightly after I used GridSearchCV to find 
# the best parameters, which is kind of troubling. But let's go on. 

# I started playing around with pipielines here to tweak my my parameters and 
# increase recall. Originally, my precisions was .69421 and recall was .08400

from sklearn import preprocessing


skb = SelectKBest()

scaler = preprocessing.MinMaxScaler()

knc = KNeighborsClassifier()

pipe = Pipeline(steps=[('scaling', scaler), ("SKB", skb), ("KNeighborsClassifier", 
                       knc)])
    
#clf.fit(features_train, labels_train)    
    
parameters = {'SKB__k': range(1,7), 'KNeighborsClassifier__n_neighbors': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45], 
              'KNeighborsClassifier__weights': ('distance', 'uniform')}

CV_SSS = StratifiedShuffleSplit(labels, 100, random_state=42)

knc_clf = grid_search.GridSearchCV(pipe, parameters, scoring = "f1", cv=CV_SSS)

knc_clf.fit(features, labels)

clf = knc_clf.best_estimator_


    
# Let's Use the Tester Script to Find the Precision and Accuracy of this
# Classifier

    
from tester import test_classifier

test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)