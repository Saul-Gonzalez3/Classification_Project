# <a name="top"></a>Classification_Project

by Saul Gonzalez
 

[Project Plan](#Project_Plan) | [Data Dictionary](#Data_Dictionary) | [Conclusions](#Conclusions) | [Next Steps](#Next_Steps) | [Recommendations](#Recommendations) | [Steps to Reproduce My Work](#Steps_to_Reproduce_My_Work)|

***
<h3><b>Project Description:</b></h3>  

This project contains the findings of research derived from the utilization of classification machine learning models paired with feature selection to determine the highest drivers that predict customer churn. The data obtained for this research was acquired from Codeup's SQL database.

    
***
<h3><b>Project Goal:</b></h3>  

Find drivers for customer churn at Telco. Why are customers churning?

Construct a ML classification model that accurately predicts customer churn

Present your process and findings to the lead data scientist


***
<h4><b>Initial Questions:</b></h4>

1. Is <b>tenure</b> associated with <b>monthly charges</b> as a driver of churn? 

2. Does <b>contract type</b> significantly affect churn?

3. Does the feature <b>partner</b> significantly affects churn?

4. Does the feature <b>dependents</b> significantly affect churn?

***
<a name="Project_Plan"></a><h3><b>Project Plan:</b></h3>

1. Create all the files I will need to make a funcitoning project (.py and .ipynb files).

2. Create a .gitignore file and ignore my env.py file.

3. Start by acquiring data from the codeup database and document all my initial acquisition steps in the acquire.py file.

4. Using the prepare file, clearn the data and split it into train, validatate, and test sets.

5. Explore the data. (Focus on the main main questions)

6. Answer all the questions with statistical testing.

7. Identify drivers of churn. Make prediction of customer churn using driving features of churn.

8. Document findings (include 4 visuals)

9. Add important finding to the final notebook.

10. Create csv file of test predictions on best performing model.

[[Back to top](#top)]

***
<a name="Data_Dictionary"></a><h3><b>Data Dictionary:</b></h3>

Feature	Definition
|**Feature**|**Definition**|
|----|----|
|`payment_type_id`| Represents an identifier or code for different payment types, such as credit card, bank transfer, or electronic payment methods.|
|`gender`| Represent the gender of the customer, indicated as "Male" or "Female".|
|`senior_citizen`| Indicate whether the customer is classified as a senior citizen, represented as a binary variable (1 for senior citizen, 0 for non-senior).|
|`partner`| Indicate whether the customer has a partner or spouse, represented as a binary variable (1 for yes, 0 for no).|
|`dependents`| Indicate whether the customer has dependents (e.g., children or other family members), represented as a binary variable (1 for yes, 0 for no).|
|`tenure`| Represents the length of time (in months) that a customer has been using the service or has been a subscriber.|
|`phone_service`| This column could indicate whether the customer has phone service, represented as a binary variable (1 for yes, 0 for no)."
|`tech_support`| Indicates whether the customer has technical support available for their service (e.g., 1 for yes, 0 for no).|
|`streaming_tv`| Indicates whether the customer has access to streaming television services (e.g., 1 for yes, 0 for no).|
|`streaming_movies`| Indicates whether the customer has access to streaming movie services (e.g., 1 for yes, 0 for no).|
|`paperless_billing`| Indicates whether the customer has opted for paperless billing (e.g., 1 for yes, 0 for no).|
|`monthly_charges`| Represents the monthly charges or fees for the service subscribed by the customer.|
|`total_charges`| Represents the total charges incurred by the customer during their tenure.|
|`churn`| Indicates whether the customer has churned or discontinued their service (e.g., 1 for churned, 0 for active).|
|`contract_type`| Represents the type of contract or service agreement (e.g., month-to-month, one-year contract, two-year contract).|
|`internet_service_type`| Represents the type of internet service (e.g., DSL, fiber optic, cable).|
|`payment_type`| Represents the payment method used by the customer (e.g., credit card, bank transfer, electronic payment).|

[[Back to top](#top)]


***
<a name="Conclusions"></a><h3><b>Conclusions:</b></h3>

Overall, my initial look at the data provided led me to ask questions regarding how the individual features were potentially affecting the target (churn). I  did an exploratory look at the data and found that there could be any number of features that had some impact on churn. After comparing each feature against <b>monthly charges</b>, hued against <b>churn</b> for insight, I then looked at a correlation heatmap to identify the top 10 features that correlated to churn to use in my modeling process. This helped in creating post-exploration questions to which I conducted statistical testing to confirm a relationship between the feature and churn, which the testing proved all features had relationships.  

Before the modeling process, I extracted the top 5 features pertaining to the target <b>churn</b> using KBest Select and Recursive Feature Engineering (RFE) to see what their recommended top 5 features were. I then fed the heatmap features, KBest Select and RFE features each in my following classification models.

Heatmap Features:
- internet_service_type_Fiber optic
- payment_type_Electronic check
- monthly_charges_scaled
- paperless_billing_encoded
- paperless_billing_Yes
- senior_citizen
- multiple_lines_Yes
- streaming_movies_Yes
- streaming_tv_Yes
- phone_service_Yes

KBest Select Features:
- tenure_scaled
- contract_type_Two year
- internet_service_type_Fiber optic
- internet_service_type_None
- payment_type_Electronic check

RFE Features:
- dependents_encoded
- phone_service_encoded
- dependents_Yes
- online_security_No internet service
- streaming_movies_No internet service

Classification Models Used:
- Decision Tree
- K Nearest Neighborns (KNN)
- Random Forest

The baseline accuracy was determined to be 73%, so my goal was to find the best set of features with the best model to predict churn.

The Best Model and set of features found was the Random Forest Model using SelectK Best Top 5 features.

The results:

Accuracy of random forest classifier on Training set: 0.81
Accuracy of random forest classifier on Validate set: 0.78
Accuracy of random forest classifier on Test set: 0.79

Ultimately, my Random Forest Model beat the baseline by 6% and the biggest driving features of churn were tenure_scaled, contract_type_Two year, internet_service_type_Fiber optic, internet_service_type_None, and payment_type_Electronic check.

***    
<a name="Next_Steps"></a><h3><b>Next Steps:</b></h3>

As a Data Scientist, if given more time to work on this project, I would dig further into ALL of my individual features, conducting more than just the chi-tests and I'd also include other models in my testing to see what comes out of it.

    
[[Back to top](#top)]
    

***    
<a name="Recommendations"></a><h3><b>Recommendations:</b></h3>  

I would definitely recommend further evaluation and consideration into the one and two-year contracts currently provided to customers and find a way to retain those who have low tenure. A possible solution could be giving satiable incentives that reduce a customers monthly bill the longer a customer stays.
    
[[Back to top](#top)]
    

***    
<a name="Steps_to_Reproduce_My_Work"></a><h3><b>Steps to Reproduce My Work:</b></h3>

1. Clone this repo.

2. Acquire the data from the Codeup's SQL Dataset.

3. Put the data in the file containing the cloned repo.

4. Run your notebook.
    
[[Back to top](#top)]