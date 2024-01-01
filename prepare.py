import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import acquire


telco_query = """
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN payment_types USING (payment_type_id)
        """
#------------------------------------------------------------------------------
def prep_telco(telco) -> pd.DataFrame:
    '''
    prep_telco will take in a a single pandas dataframe
    presumed of the same structure as presented from 
    the acquire module's get_telco_data function (refer to acquire docs)
    returns a single pandas dataframe with redudant columns
    removed and missing values filled.
    '''
    telco = telco.drop(
    columns=[
        'Unnamed: 0',
        'internet_service_type_id',
        'payment_type_id',
        'contract_type_id',   
    ])
    telco.loc[:,'internet_service_type'] = telco.internet_service_type.\
    fillna('no internet')
    telco = telco.set_index('customer_id')
    telco.loc[:,'total_charges'] = (telco.total_charges + '0')
    telco.total_charges = telco.total_charges.astype(float)
    return telco


#------------------------------------------------------------------------------
def wrangle_data(dataset=None):
    wrangles = {
        
        'telco': prep_telco(acquire.get_telco_data(telco_query, './')), 
    }
    if dataset:
        return wrangles[dataset]
    else:
        print('please ask to get something ok')

#------------------------------------------------------------------------------
def split_data(df, dataset=None):
    target_cols = {
        'telco': 'churn',
    }
    if dataset:
        if dataset not in target_cols.keys():
            print('please choose a real dataset tho')

        else:
            target = target_cols[dataset]
            train_val, test = train_test_split(
                df,
                train_size=0.8,
                stratify=df[target],
                random_state=1349)
            train, val = train_test_split(
                train_val,
                train_size=0.7,
                stratify=train_val[target],
                random_state=1349)
            return train, val, test
    else:
        print('please specify what df we are splitting.')
        
        