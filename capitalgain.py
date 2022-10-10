from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import io
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Open the source data CVS file
adultdf = pd.read_csv(
    r'\adult.csv', header=None, sep=',\s', engine='python')

# Assign the title to columns
title = ['Age', 'Work_Group', 'Id', 'Education', 'Education_No', 'Marital_Status', 'Job',
         'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Weekly_Work', 'Country', 'Income']
adultdf.columns = title

# Drop unrelated columns
adultdf.drop(['Id', 'Education', 'Capital_Loss'], axis=1, inplace=True)

# Assign 1 for existence of Capital Gain and 0 for non
adultdf['Capital_Gain'] = (adultdf['Capital_Gain'] > 0).astype(int)

# Finding categorical variables
categoricalvar = [
    variables for variables in adultdf.columns if adultdf[variables].dtype == 'O']

# Checking for missing items in categorical variables
print('\nMissing items in categorical variables:\n',
      adultdf[categoricalvar].isnull().sum())

# View frequency of categorical variables
for catvar in categoricalvar:
    print('\nFrequency of categorical variables:\n',
          adultdf[catvar].value_counts())

# Replace ? with NaN in
adultdf = adultdf.replace('?', np.NaN)

# Fill NaN values with mode
adultdf = adultdf.fillna(adultdf.mode().iloc[0])

# Check missing values in numerical Categories
numericalcat = [cat for cat in adultdf.columns if adultdf[cat].dtype != 'O']
print('\nMissing items in Numerical variables:\n',
      adultdf[numericalcat].isnull().sum())

# Define feature vectore and target category
X = adultdf.drop(['Capital_Gain'], axis=1)
y = adultdf['Capital_Gain']


# Create Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']


# Encode categorical variables with OneHot encoding
encoder = ce.OneHotEncoder(cols=['Work_Group', 'Marital_Status', 'Job', 'Relationship',
                                 'Race', 'Sex', 'Country', 'Income'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Feature Scaling
cols = X_train.columns
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

# Load 5 machine learining models
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
gnb = GaussianNB()
abc = AdaBoostClassifier()
lrg = LogisticRegression(max_iter=1500, fit_intercept=False, penalty='none')
# after loading 2 models increase max_iter required to avoid error


# RandomForestClassifier
print('\n =============  RandomForest Classifier result  ================ \n')
rfc.fit(X_train, y_train)


y_pred = rfc.predict(X_test)
print('RandomForest Classifier accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))
y_pred_train = rfc.predict(X_train)
print(
    'RandomForest Classifier Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('RandomForest Classifierr set score: {:.4f}'.format(
    rfc.score(X_train, y_train)))
print('RandomForest Classifier Test set score: {:.4f}'.format(
    rfc.score(X_test, y_test)))

# Class distribution in test set
print(y_test.value_counts())

# Null accuracy score
null_accuracy = (8963/(8963+806))
print('RandomForest Classifier Null accuracy score: {0:0.4f}'. format(
    null_accuracy))

# Print the Confusion Matrix
cmrfc = confusion_matrix(y_test, y_pred)
print('\nRandomForest Classifier Confusion matrix:\n', cmrfc)
print('\nTrue Positives(TP) = ', cmrfc[0, 0])
print('\nTrue Negatives(TN) = ', cmrfc[1, 1])
print('\nFalse Positives(FP) = ', cmrfc[0, 1])
print('\nFalse Negatives(FN) = ', cmrfc[1, 0])

# Print Classification Report
print(classification_report(y_test, y_pred))

# Calculate classification accuracy
TP = cmrfc[0, 0]
TN = cmrfc[1, 1]
FP = cmrfc[0, 1]
FN = cmrfc[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('RandomForest Classifier Classification accuracy : {0:0.4f}'.format(
    classification_accuracy))

# Calculate classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('RandomForest Classifierr Classification error : {0:0.4f}'.format(
    classification_error))

# Calculate Precision score
precision = TP / float(TP + FP)
print('RandomForest Classifierr Precision : {0:0.4f}'.format(precision))

# Calculate Recall
recall = TP / float(TP + FN)
print('RandomForest Classifierr Recall : {0:0.4f}'.format(recall))

# Applying 10-Fold Cross Validation
scores = cross_val_score(rfc, X_train, y_train, cv=10, scoring='accuracy')
print('\nDRandomForest Classifier Cross-validation scores:\n{}'.format(scores))
print(
    '\nRandomForest Classifier Average cross-validation score: {:.4f}'.format(scores.mean()))


# DecisionTree Classifier
print('\n =============  DecisionTree Classifier result  ================ \n')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print('DecisionTree Classifier accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))
y_pred_train = dtc.predict(X_train)
print(
    'DecisionTree Classifier Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('DecisionTree Classifier set score: {:.4f}'.format(
    dtc.score(X_train, y_train)))
print('DecisionTree Classifier Test set score: {:.4f}'.format(
    dtc.score(X_test, y_test)))

# Class distribution in test set
print(y_test.value_counts())

# Calculate Null accuracy score
null_accuracy = (8963/(8963+806))
print('DecisionTree Classifier Null accuracy score: {0:0.4f}'. format(
    null_accuracy))

# Print the Confusion Matrix
cmdtc = confusion_matrix(y_test, y_pred)
print('\nDecisionTree Classifier Confusion matrix:\n', cmdtc)
print('\nTrue Positives(TP) = ', cmdtc[0, 0])
print('\nTrue Negatives(TN) = ', cmdtc[1, 1])
print('\nFalse Positives(FP) = ', cmdtc[0, 1])
print('\nFalse Negatives(FN) = ', cmdtc[1, 0])

# Classification Report
print(classification_report(y_test, y_pred))

# Claculate classification accuracy
TP = cmdtc[0, 0]
TN = cmdtc[1, 1]
FP = cmdtc[0, 1]
FN = cmdtc[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('DecisionTree Classifier Classification accuracy : {0:0.4f}'.format(
    classification_accuracy))

# Calculate classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('DecisionTree Classifier Classification error : {0:0.4f}'.format(
    classification_error))

# Calculate Precision score
precision = TP / float(TP + FP)
print('DecisionTree Classifier Precision : {0:0.4f}'.format(precision))

# Calculate Recall
recall = TP / float(TP + FN)
print('DecisionTree Classifier Recall : {0:0.4f}'.format(recall))

# Applying 10-Fold Cross Validation
scores = cross_val_score(dtc, X_train, y_train, cv=10, scoring='accuracy')
print('\nDecisionTree Classifier Cross-validation scores:\n{}'.format(scores))
print(
    '\nDecisionTree Classifier Average cross-validation score: {:.4f}'.format(scores.mean()))


# GaussianNB Model
print('\n =============  GaussianNB model result  ================ \n')
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('GaussianNB Model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)
print(
    'GaussianNB Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('GaussianNB Training set score: {:.4f}'.format(
    gnb.score(X_train, y_train)))
print('GaussianNB Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

# check class distribution in test set
print(y_test.value_counts())

# Calculate null accuracy score
null_accuracy = (8963/(8963+806))
print('GaussianNB Null accuracy score: {0:0.4f}'. format(null_accuracy))

# Print the Confusion Matrix
cmgnb = confusion_matrix(y_test, y_pred)
print('\nGaussianNB Confusion matrix:\n', cmgnb)
print('\nTrue Positives(TP) = ', cmgnb[0, 0])
print('\nTrue Negatives(TN) = ', cmgnb[1, 1])
print('\nFalse Positives(FP) = ', cmgnb[0, 1])
print('\nFalse Negatives(FN) = ', cmgnb[1, 0])

# Classification Report
print(classification_report(y_test, y_pred))

# Classification accuracy
TP = cmgnb[0, 0]
TN = cmgnb[1, 1]
FP = cmgnb[0, 1]
FN = cmgnb[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('GaussianNB Classification accuracy : {0:0.4f}'.format(
    classification_accuracy))

# Claculate classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('GaussianNB Classification error : {0:0.4f}'.format(
    classification_error))

# Calculate Precision score
precision = TP / float(TP + FP)
print('GaussianNB Precision : {0:0.4f}'.format(precision))

# Calculate Recall
recall = TP / float(TP + FN)
print('GaussianNB Recall : {0:0.4f}'.format(recall))

# Applying 10-Fold Cross Validation
scores = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy')
print('\nGaussianNB Cross-validation scores:\n{}'.format(scores))
print(
    '\nGaussianNB Average cross-validation score: {:.4f}'.format(scores.mean()))


print('\n =============  AdaBoost classifier model result  ================ \n')
# fit the model GaussianNB
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
print('AdaBoost classifier Model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))
y_pred_train = abc.predict(X_train)
print(
    'AdaBoost classifier Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('AdaBoost classifier Training set score: {:.4f}'.format(
    abc.score(X_train, y_train)))
print('AdaBoost classifier Test set score: {:.4f}'.format(
    abc.score(X_test, y_test)))

# check class distribution in test set
print(y_test.value_counts())

# Calculate null accuracy score
null_accuracy = (8963/(8963+806))
print('AdaBoost classifier Null accuracy score: {0:0.4f}'. format(
    null_accuracy))

# Print the Confusion Matrix
cmabc = confusion_matrix(y_test, y_pred)
print('\nAdaBoost classifier Confusion matrix:\n', cmabc)
print('\nTrue Positives(TP) = ', cmabc[0, 0])
print('\nTrue Negatives(TN) = ', cmabc[1, 1])
print('\nFalse Positives(FP) = ', cmabc[0, 1])
print('\nFalse Negatives(FN) = ', cmabc[1, 0])

# Classification Report
print(classification_report(y_test, y_pred))

# Calculate classification accuracy
TP = cmabc[0, 0]
TN = cmabc[1, 1]
FP = cmabc[0, 1]
FN = cmabc[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('AdaBoost classifier Classification accuracy : {0:0.4f}'.format(
    classification_accuracy))

# Calculate classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('AdaBoost classifier Classification error : {0:0.4f}'.format(
    classification_error))

# Calculate Precision score
precision = TP / float(TP + FP)
print('AdaBoost classifier Precision : {0:0.4f}'.format(precision))

# Calculate Recall
recall = TP / float(TP + FN)
print('AdaBoost classifier Recall : {0:0.4f}'.format(recall))

# Applying 10-Fold Cross Validation
scores = cross_val_score(abc, X_train, y_train, cv=10, scoring='accuracy')
print('\nAdaBoost classifier Cross-validation scores:\n{}'.format(scores))
print(
    '\nAdaBoost classifier Average cross-validation score: {:.4f}'.format(scores.mean()))

print('\n =============  Logistic Regression classifier model result  ================ \n')
# fit the model GaussianNB
lrg.fit(X_train, y_train)
print(lrg.coef_)
y_pred = lrg.predict(X_test)
print('Logistic Regression Model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))
y_pred_train = lrg.predict(X_train)
print(
    'Logistic Regression Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('Logistic Regression Training set score: {:.4f}'.format(
    lrg.score(X_train, y_train)))
print('Logistic Regression Test set score: {:.4f}'.format(
    lrg.score(X_test, y_test)))

# check class distribution in test set
print(y_test.value_counts())

# Calculate null accuracy score
null_accuracy = (8963/(8963+806))
print('Logistic Regression Null accuracy score: {0:0.4f}'. format(
    null_accuracy))

# Print the Confusion Matrix
cmlrg = confusion_matrix(y_test, y_pred)
print('\nLogistic Regression Confusion matrix:\n', cmlrg)
print('\nTrue Positives(TP) = ', cmlrg[0, 0])
print('\nTrue Negatives(TN) = ', cmlrg[1, 1])
print('\nFalse Positives(FP) = ', cmlrg[0, 1])
print('\nFalse Negatives(FN) = ', cmlrg[1, 0])


# Print Classification Report
print(classification_report(y_test, y_pred))

# print classification accuracy
TP = cmlrg[0, 0]
TN = cmlrg[1, 1]
FP = cmlrg[0, 1]
FN = cmlrg[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Logistic Regression accuracy : {0:0.4f}'.format(
    classification_accuracy))

# print classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Logistic Regression Classification error : {0:0.4f}'.format(
    classification_error))

# Calculate Precision score
precision = TP / float(TP + FP)
print('Logistic Regression Precision : {0:0.4f}'.format(precision))

# Calculate Recall
recall = TP / float(TP + FN)
print('Logistic Regression Recall : {0:0.4f}'.format(recall))


# Applying 10-Fold Cross Validation
scores = cross_val_score(lrg, X_train, y_train, cv=10, scoring='accuracy')
print('\nLogistic Regression Cross-validation scores:\n{}'.format(scores))
print(
    '\nLogistic Regression Average cross-validation score: {:.4f}'.format(scores.mean()))
