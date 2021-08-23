# credit_risk_analysis

## Overview

LendingClub, a peer-to-peer lending company plans to predict credit risk using machine learning to provide a quicker, more reliable loan experience.  Predicting good loan candidates should lead to lower default rates.  

Several machine learning models from the imbalanced-learn and scikit-learn libraries were used to predict credit risk.  Each model is evaluated by calculating the balanced accuracy score, a confusion matrix, and printing a classification report.

## Results
The data provided through LendingClub was imported and cleaned, and saved as a Pandas DataFrame.  Null rows and columns were dropped and strings were converted to numerical data for evaluation.  The data was separated into 90 features with the target variable “loan status” valued as **low_risk** (shown as 1) or **high_risk** (shown as 0).  The function **train_test_split** was then used to split the data into training and testing sets.  Several models were tested, and the results are as follows:

     * Random Oversampling results: 
![RandomOversampling](/images/RandomOversampling.png)
A mediocre accuracy score of 0.6468 leaves room for improvement in this model.
High-risk recall: 0.73 
Low-risk recall: 0.56

     * SMOTE Oversampling results:
![SMOTEOversampling](/images/SMOTEOversampling.png)
A mediocre accuracy score of 0.6542 leaves room for improvement in this model.
High-risk recall: 0.64
Low-risk recall: 0.66  

     * Cluster Centroids Undersampling results:
![ClusterCentroids](/images/ClusterCentroids.png)
A low accuracy score of 0.5348 along with the Low-risk recall of 0.40 (this is the lowest recall for the low-risk group of all the models tested here) indicates this is the least predictive model in this study.

     * Combination SMOTEENN results:
![SMOTEENN](/images/SMOTEENN.png)
A mediocre accuracy score of 0.6472 leaves room for improvement in this model.
High-risk recall: 0.73 
Low-risk recall: 0.56

     * Balanced Random Forest results:
![BRF](/images/BRF.png)
A high accuracy score of 0.7064, however, the sensitivity of the high-risk group is one of the lowest indicating the algorithm incorrectly predicts false negatives for the high-risk group.
High-risk recall: 0.59 
Low-risk recall: 0.82

     * EasyEnsemble AdaBoost results:
![AdaBoost](/images/AdaBoost.png)
Of all the models tested, the Adaptive Boosting (AdaBoost) method captured the highest accuracy score at 0.7334.
High-risk recall: 0.70
Low-risk recall: 0.76

   

## Summary
Based on accuracy scores alone, the Adaptive Boosting method appears to be the best suited for successfully predicting the low-risk candidates.  However, in this case, we not only need to predict good low-risk candidates, but should identify the high-risk candidates without incorrectly identifying them as low-risk and so we consider the high-risk recall value an important part of this study.  Based on the high (and notably balanced) recall scores for the Adaptive Boosting method, it appears that this algorithm is best suited for predicting credit risk.  

Note that while the Random Oversampling and SMOTEENN models have fair accuracy scores and both demonstrate the highest recall rates for the High-risk recall group, both of these models also have low recall for the low-risk group indicating the models’ tendency to incorrectly deny loans to low-risk candidates.
