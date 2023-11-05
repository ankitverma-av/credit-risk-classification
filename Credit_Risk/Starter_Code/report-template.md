#   Report 

## Overview of the Analysis

Purpose of the analysis.

    Lending institutions provide loans or assets to borrowers with the anticipation that the borrower will either return the asset or repay     the loan. Credit Risk arises when a borrower fails to return the asset or fulfill the loan repayment, resulting in financial losses for     the lender. Lenders assess this risk through various methods, but in this study, we will employ Machine Learning to examine a dataset of     past lending transactions from a peer-to-peer lending platform. The objective is to develop a model that can assess the creditworthiness     of borrowers.


* Explain what financial information the data was on, and what you needed to predict.

    The dataset included some financial variables in a CSV file (lending_data.csv). The provided financial information included independant     variables(X) such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks on a credit       file, and total debt. The dependent variable (y) was loan_status, which indicates if a loan is healthy or high-risk.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
    The dependent variable (loan_status), shows two segments: 0 for healthy loans and 1 for high-risk loans. The value_counts indicated a       significantly higher amount of healthy loans vs. high-risk ones.
    
    0    75036
    1     2500
    Name: loan_status, dtype: int64


* Describe the stages of the machine learning process you went through as part of this analysis.
    - After reading the data the first step is to seperating the data into labels and features i.e X, y
    - Splitting the data into training and testing datasets
    - Fitting a logistic regression model by using the training data
    - Evaluation and prediction : The model was used to predict loan status on the testing set, and then it was evaluated using                   accuracy score , confusion matrix, and classification report
    - Oversampling: To improve accuracy, RandomOverSampler was used.
    - Model re-fit and evaluation: A new model was trained on the resampled dataset to compare against the original model.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
    To improve accuracy and enhance the model's ability to classify non-healthy loans, we uses the RandomOverSampler module from the             imbalanced-learn library. This technique involves adding more copies of the minority class (non-healthy loans) to create a balanced         dataset.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
     
                      precision    recall  f1-score   support

  0 (Healthy loan)       1.00      0.99      1.00     18765
1 (High-Risk loan)       0.85      0.91      0.88       619

          accuracy                           0.99     19384
         macro avg       0.92      0.95      0.94     19384
      weighted avg       0.99      0.99      0.99     19384
  
  Accuracy: is how often the model is correctly  predicting to the total number of observations
  Precison : is the ratio of correctly predicted positive observations to the total predicted positive observations
  Recall : is the ratio of correctly predictive positive observations to all  predicted observation s for that class.

  > In comparison to the original dataset, similarly by looking at support numbers we can see the number of healthy loans is greater than       the number of unhealthy loans which means the data is imbalanced
  > The model has a good accuracy score of 99%, the precision score for 0 (healthy loans) is 100% and the precision for High-Risk label is       not bad at 85% however 15% of wrong predictions could lead to missing high-risk loans. The model should continue to be trained with         more data to improve its Precision.
  > The recall score is also quite high at 99% for prediction of Healthy loan labels and 91% for high-risk loans label 1.

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
                      precision    recall  f1-score   support

  0 (Healthy loan)       0.99      0.99      0.99     75036
1 (High-Risk loan)       0.99      0.99      0.99     75036

          accuracy                           0.99    150072
         macro avg       0.99      0.99      0.99    150072
      weighted avg       0.99      0.99      0.99    150072

    
  
   > The accuracy score for this model is also quite high at 99%. Looking at the confusion matrix, the oversampled data model did                 significantly better at predicting false negatives and true positives
   > The precision score which is true negative for healthy loans was 100% before where as with oversampled data it is 99% however both the       models predicted same false positives at 99% recall score 
   > The precision and recall score improved for high-risk loans compared to the previous model.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?

- The original Regression model (train/test data), showed a high accuracy rate. However, it showed a proportion of high-risk loans that         could potentially be wrongly classified. The model with the oversampled data showed an improvement in predicting high-risk loans. While    there was a sligt increase in false positives (healthy loans predicted as high-risk) this could be more acceptable. The key performance      metrics, such as the balanced accuracy score and the F1 score for high-risk loans, were better compared to the original model.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

In this context (loan risk prediction), it's crucial to minimize false negatives (predicting a high-risk loan as healthy) as this could result in financial losses, whereas a false positive (predicting a healthy loan as high-risk) might result in missed opportunities. Considering this, Precision, Recall, and F1 score for high-risk loans are particularly important.


