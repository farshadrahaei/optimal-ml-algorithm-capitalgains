Title

Predict earning “Capital Gains” on census income data with optimal machine learning algorithm.

Abstract

Capital Gains are profits resulted from any sort of investment; in this project I am going to predict if the individual got “Capital Gains” based on census income data.
Data has been trained in python with multiple machine learning algorithms and based on outcome on different models, “Logistic Regression” has been identified as the algorithm that has better performance for this experiment.
Training outcomes also prove that census income data like Age, Job, Education, Marital Status, Race, Sex, Country, and Income are useful to predict earning of capital gains for individual.

Introduction

Being able to predict individual financial situation like income, capital gain/loose and tax obligation been always a challenging matter, as a lot of different factors could affect the outcome.
The result of this sort of prediction could be used by governments, finance institute to adjust their policies, products, and services to better serve people.
Capital gains are profits resulted from any sort of investment that trigger tax liability for individual, which make it a desirable item to be predicted.
Using machine learning algorithm and training data, I am going to predict earning “Capital Gains” for individual based on census data.

Related Work

A brief look at Kaggle website we can find out similar experiments trying to predict people “Income” using different type of data including “US Household income data” and “Census data”. Here are some links for your reference:

https://www.kaggle.com/goldenoakresearch/us-household-income-stats-geo-locations

https://www.kaggle.com/uciml/adult-census-income

In this experiment I focused on “Capital Gains” prediction which is not being addressed by other similar works.
 
Approach 

Hypothesis

Null Hypothesis H0:

Census income data does not affect earning Capital Gains.

Alternative Hypothesis HA:

Census income data (Age, Job, Education, Marital Status, Race, Sex, Country, Income)
affect possibility of earning Capital Gains.

Design of experiment

To be able to test the hypothesis first the data must be prepared with some operations, once the data is ready it could be loaded to the python and trained by machine learning algorithm.
Using the model’s outcome efficacy of the algorithm will be identified and hypothesis could be accepted or rejected.
Below implementation steps has been explained:

Data Preparation steps

1) Loading libraries to python.
2) Importing CSV source data file to python.
3) Data file did not have any titles, adding titles to data columns.
4) Drop unrelated columns for better accuracy.
5) Assign 1 for existence of Capital Gain and 0 for nonexistence.
6) Check the missing values in categorical variables.
7) Check all the categorical variables if they assigned with wrong values.
8) Substitute ‘?’ with ‘NaN’.
9) Fill NaN values with mode.
10) Check the numerical categories for issues.

Training data and generating results steps

1) Create train and test sets.
2) Encode categorial variables.
3) Feature scaling.
4)  Load 5 different models to compare the performance:
	- DecisionTree Classifier
	- RandomForest Classifier
- GaussianNB
- AdaBoostClassifier
- LogisticRegression
5) For each of 3 models:
- Predict the result
- Check the accuracy
- Generate Confusion matrix and identify TP, TN, FP, FN
- Compare model accuracy with null accuracy
- Generate Classification report, classification accuracy, classification error, precision and recall
- Apply 10-Fold Cross Validation and calculate average cross validation

Issues

Having 5 trainings model in the code caused “TOTAL NO. of ITERATIONS REACHED LIMIT” error, to fix the issue “max_iter” has been increased. 

Experimental Evaluation 

Source Data

This dataset provided in Kaggle website:

https://www.kaggle.com/qizarafzaal/adult-dataset
And include bellow data:

'Age', 'Work_Group', 'Id', 'Education', 'Education_No', 'Marital_Status', 'Job', 'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Weekly_Work', 'Country', 'Income'.

Hardware

PC computer with core i5 cpu and 8GB of RAMs.

Software
-	OS Windows 10
-	Python 3.9.4

Interpretation of Results

Compare Models

	Decision Tree Classifier	Random Forest Classifier	Gaussian NB	AdaBoost Classifier	Logistic Regression	Best Performance
Accuracy	0.8581	0.9023	0.1163	0.9171	0.9174	LogisticRegression
Error	0.1419	0.0977	0.8837	0.0829	0.0826	LogisticRegression
Precision	0.9215	0.9780	0.0390	0.9991	0.9999	LogisticRegression
Recall	0.9237	0.9205	0.9459	0.9178	0.9175	GaussianNB
10 folds Ave	0.8554	0.9003	0.1160	0.9164	0.9163	AdaBoostClassifier

Considering above chart the highest accuracy and lowest Error together with highest precision are related to Logistic Regression Model, we see also Gaussian NB has the highest recall and AdaBoost Classifier highest 10 folds average.
Overall Logistic Regression following by AdaBoost Classifier models provided the best performance.

Analyze Logistic Regression Model Results

-	“Training set score = 0.9164” and “Test set score = 0.9174” which are very close numbers so there is not over or under fitting.
-	Comparing “Model accuracy = 0.9174” and “Null Accuracy =0.9175” which very close prove the model is in good shape and trustworthy.
-	High “Classification accuracy= 0.9174” and low “classification error= 0.0826” also prove the model is good.
-	“10 folds Average cross-validation = 0.9163” means the model is accurate on average about %91.63.
-	There are some small differences between each fold cross validations, the folds cross validation is between 0.9161 to 0.9166 that means model is not dependent on any particular fold.
-	True Positives (TP) is actual correct prediction of existence of observation within a certain right class equal to 8962, which the result is an ideal number.
-	True Negatives (TN) is actual correct prediction of not existence of observation within a certain right class equal to 0, which the result is not a good number.
-	False Positives (FP) is wrong prediction of existence of observation within a certain class equal to 1, which the result is a good number.
-	False Negatives (FN) is wrong prediction of not existence of observation within a certain class equal to 806, which the result is not looking ideal. Considering recall score of 0.9175 which is close to 1 we can conclude the number of False Negative is negligible.

Other attempts to improve performance

-	“Capital_Gain” column consist of 0 for existence of “No Capital Gain” and 1 for existence of “Capital Gain”, which is int type. In other attempt I converted the column to obj and assign “Capital_Gain” and “No_Capital_Gain” to it. The result was the same with no improvement. 

-	To get more familiar with data set prediction results and make sure data cleanup has been done correctly, I also predicted the “Income”. In this case “AdaBoostClassifier” provided the best performance.

Features and Algorithms overview

Features

Age, Job, Education, Marital Status, Race, Sex, Country, Income are the individual independent values in data used in this model to predict earning Capital Gains. 
These features included numerical and categorical values. Using Onehot coding, categorical features converted to one feature per category, each binary.
There were some other unrelated features included in this data set like person id that has been removed from training and model.
Logistic Regression
Logistic regression is used when the outcome is a discrete variable. Example, trying to figure out who will win the election, whether a student will pass or fail an exam, whether a customer will come back, whether an email is a spam. This is commonly called as a classification problem because we are trying to determine which class the data set best fits.
Which is in our case is one of the best candidates to model and predict earning capital gains, considering the model outcome values including accuracy of 0.9174 and error rate of 0.0826 it proofs to be the best model.
Adaboost
AdaBoost training process selects only those features known to improve the predictive power of the model, reducing dimensionality and potentially improving execution time as irrelevant features don't need to be computed.
AdaBoost can be used to boost the performance of any machine learning algorithm. It is best used with weak learners. These are models that achieve accuracy just above random chance on a classification problem.
Considering only 8 features and the total size of our data and the fact that this model is not suffering from dimensionality, adaboost might not be the best option. Training results shows yet it could provide acceptable result and got the second rank after logistic regression with the best 10 folds ave. of 0.9164. 
Decision Tree
A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The paths from root to leaf represent classification rules.
In decision analysis, a decision tree and the closely related influence diagram are used as a visual and analytical decision support tool, where the expected values of competing alternatives are calculated.
Considering unrelated features that does not necessarily could make the branches of the tree this algorithm will likely not be ideal for this model, training result also confirmed high error rate of 0.1419.
Random Forest
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean/average prediction of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees.
Since the features are not good candidate to make the branches of tree this model is not also an ideal candidate, but training data shows slightly improvement on error of 0.0977 in compare with decision tree of 0.1419 which was expected.


Gaussian Naive Bayes

Naive Bayes is a simple technique for constructing classifiers, models that assign class labels to problem instances represented as vectors of feature values, where the class labels are drawn from some finite set.
In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature which is theory sounds to be a good candidate based on features type and description on this experiment, but the training result shows the highest rate of 0.88 which is unacceptable.

Conclusion

Based on result of comparing different machine learning algorithms using accuracy, error, precision, recall and 10 folds average criteria we realized “Logistic Regression” has better performance among other algorithms tested on this experiment.
Considering high model accuracy of 0.9174 and low error rate of 0.0826 on logistic regression model, it has been concluded this model is acceptable and could be used for prediction of “Capital Gains” based on “income census data.

References

https://ikompass.edu.sg/logistic-regression-use-case-classification-problems-by-prakash-roshan
https://en.wikipedia.org/wiki/AdaBoost
https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
https://en.wikipedia.org/wiki/Decision_tree
https://en.wikipedia.org/wiki/Random_forest
https://en.wikipedia.org/wiki/Naive_Bayes_classifier

