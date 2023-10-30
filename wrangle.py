import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import acquire
import os
import env
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import env
import acquire
import wrangle

# imports for modeling:
# import Logistic regression
from sklearn.linear_model import LogisticRegression
# import K Nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
# import Decision Trees:
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# import Random Forest:
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

# interpreting our models:
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


directory = os.getcwd()
#------------------------------------------------------------------------------
def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url to access mySQL db
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
#------------------------------------------------------------------------------
SQL_query = """
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN payment_types USING (payment_type_id)
        """
#------------------------------------------------------------------------------

telco_query = """
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN payment_types USING (payment_type_id)
        """

# ------------------- TELCO DATA -------------------

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn_encoded)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn_encoded)
    return train, validate, test
#------------------------------------------------------------------------------
def prep_telco_data(df):
    
    '''Preps the telco data and returns the data split into train, validate and test portions'''
    # Drop duplicate columns
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
      
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['gender', \
                              'partner', \
                              'dependents', \
                              'phone_service', \
                              'paperless_billing', \
                              'churn', \
                              'multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.drop(columns= ['gender', \
                     'partner', \
                     'dependents', \
                     'phone_service', \
                     'paperless_billing', \
                     'churn', \
                     'multiple_lines', \
                     'online_security', \
                     'online_backup', \
                     'device_protection', \
                     'tech_support', \
                     'streaming_tv', \
                     'streaming_movies', \
                     'contract_type', \
                     'internet_service_type', \
                     'churn_Yes', \
                     'payment_type']) 
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test
#------------------------------------------------------------------------------
def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df
#------------------------------------------------------------------------------
def get_chi_tenure(train):
    '''get results of chi-square between tenure and churn'''
    
    null = "There is no association between tenure and churn"
    alpha = 0.05
    observed6 = pd.crosstab(train.tenure, train.churn_encoded)

    chi2, p, degf, expected = stats.chi2_contingency(observed6)
    
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    if p < alpha:
        print(f'We reject H₀:{null}')
    else:
        print(f'We FAIL to reject H₀:{null}')
 #------------------------------------------------------------------------------
def get_chi_contracts(train):
    '''get results of chi-square between one year and two year contracts and churn'''
    
    null = "There is no association between one year and two year contracts and churn"
    alpha = 0.05
    observed7 = pd.crosstab(train['contract_type_One year'], train['contract_type_Two year'])

    chi2, p, degf, expected = stats.chi2_contingency(observed7)
    
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    if p < alpha:
        print(f'We reject H₀:{null}')
    else:
        print(f'We FAIL to reject H₀:{null}')
 #------------------------------------------------------------------------------
def get_chi_partner(train):
    '''get results of chi-square between churn and partner'''
    
    null = "There is no association between churn and partner"
    alpha = 0.05
    observed8 = pd.crosstab(train['partner_encoded'], train['churn_encoded'])

    chi2, p, degf, expected = stats.chi2_contingency(observed8)
    
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    if p < alpha:
        print(f'We reject H₀:{null}')
    else:
        print(f'We FAIL to reject H₀:{null}')
 #------------------------------------------------------------------------------
def get_chi_dependents(train):
    '''get results of chi-square between churn and dependents'''
    
    null = "There is no association between churn and dependents"
    alpha = 0.05
    observed9 = pd.crosstab(train['dependents_encoded'], train['churn_encoded'])

    chi2, p, degf, expected = stats.chi2_contingency(observed9)
    
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    if p < alpha:
        print(f'We reject H₀:{null}')
    else:
        print(f'We FAIL to reject H₀:{null}')
#------------------------------------------------------------------------------
def get_decision_tree():
    '''Returns the results of a decision tree model'''

    from wrangle import get_telco_data
    telco = get_telco_data()
    train, validate, test = prep_telco_data(telco)

    baseline = train.churn_encoded.mean()
    baseline_accuracy = (train.churn_encoded == 0).mean()

    X_train = train.drop(columns=['churn_encoded'])
    y_train = train.churn_encoded

    X_validate = validate.drop(columns=['churn_encoded'])
    y_validate = validate.churn_encoded

    X_test = test.drop(columns=['churn_encoded'])
    y_test = test.churn_encoded
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=123)
    
    clf = clf.fit(X_train, y_train)
    
    clf.predict(X_train)[:20]
    
    clf.score(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    
    print(f"Accuracy of Decision Tree on train data is {clf.score(X_train, y_train)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(X_validate, y_validate)}")    
#------------------------------------------------------------------------------
def get_logistic_regression():
    '''Returns the results of a logistic regression model'''

    from wrangle import get_telco_data
    telco = get_telco_data()
    train, validate, test = prep_telco_data(telco)

    baseline = train.churn_encoded.mean()
    baseline_accuracy = (train.churn_encoded == 0).mean()

    X_train = train.drop(columns=['churn_encoded'])
    y_train = train.churn_encoded

    X_validate = validate.drop(columns=['churn_encoded'])
    y_validate = validate.churn_encoded

    X_test = test.drop(columns=['churn_encoded'])
    y_test = test.churn_encoded
    
    # from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(C=1, class_weight={0:1, 1:99}, random_state=123, intercept_scaling=1, solver='lbfgs')
    logit2 = LogisticRegression(C=1, class_weight={0:1, 1:99}, random_state=123, intercept_scaling=1, solver='lbfgs')

    logit.fit(X_train, y_train)
    logit2.fit(X_validate, y_validate)
    
    LogisticRegression(C=1, class_weight={0: 1, 1: 99}, random_state=123)
    
    y_pred = logit.predict(X_train)
    y_pred2 = logit2.predict(X_validate)
    
    y_pred_proba = logit.predict_proba(X_train)
    y_pred_proba2 = logit2.predict_proba(X_validate)
    
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'
     .format(logit2.score(X_validate, y_validate)))
#------------------------------------------------------------------------------
def get_random_forest():
    '''Returns the results of a random forest model'''

    from wrangle import get_telco_data
    telco = get_telco_data()
    train, validate, test = prep_telco_data(telco)

    baseline = train.churn_encoded.mean()
    baseline_accuracy = (train.churn_encoded == 0).mean()

    X_train = train.drop(columns=['churn_encoded'])
    y_train = train.churn_encoded

    X_validate = validate.drop(columns=['churn_encoded'])
    y_validate = validate.churn_encoded

    X_test = test.drop(columns=['churn_encoded'])
    y_test = test.churn_encoded
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)
    
    rf2 = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)
    
    rf.fit(X_train, y_train)
    rf2.fit(X_validate, y_validate)
    
    y_pred = rf.predict(X_train)
    y_pred2 = rf.predict(X_validate)
    
    y_pred_proba = rf.predict_proba(X_train)
    y_pred_proba2 = rf.predict_proba(X_validate)
    
    print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
    print('Accuracy of random forest classifier on validate set: {:.2f}'
     .format(rf.score(X_validate, y_validate)))
#------------------------------------------------------------------------------
def get_random_forest_test():
    '''Returns the result of a random_forest model on test data'''
    from wrangle import get_telco_data
    telco = get_telco_data()
    train, validate, test = prep_telco_data(telco)

    baseline = train.churn_encoded.mean()
    baseline_accuracy = (train.churn_encoded == 0).mean()

    X_train = train.drop(columns=['churn_encoded'])
    y_train = train.churn_encoded

    X_validate = validate.drop(columns=['churn_encoded'])
    y_validate = validate.churn_encoded

    X_test = test.drop(columns=['churn_encoded'])
    y_test = test.churn_encoded
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)

    
    rf.fit(X_test, y_test)
    
    y_pred = rf.predict(X_test)
    
    y_pred_proba = rf.predict_proba(X_test)
    
    print('Accuracy of random forest classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))
