import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from xgboost import Booster


def load_model():
    with open(r'C:\Users\shirs\Desktop\CV Projects\Customer-Churn-Prediction-main\model.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

RS = data['RS']
# encoding_plan = data['encoding_plan']
ohe = data['ohe']
model = data['model']

def show_predict_page():
    st.title('Customer Churn Prediction')
    st.write("""### We need some information for prediction """)

    voice_mail_plans=('yes','no')
    voice_mail_plan=st.selectbox("Voice Mail Plans",voice_mail_plans)

    international_plans=('yes','no')
    international_plan=st.selectbox("International Plans",international_plans)

    area_codes = ['area_code_415', 'area_code_408', 'area_code_510']
    area_code=st.selectbox("Area Code",area_codes)

    account_length=st.slider("Account Length",1,243,1)
    number_vmail_messages=st.slider("Number of Voice Mail Messages",0,52,1)
    total_day_minutes=st.slider("Total minutes of day calls",0.0,351.5,1.5)
    total_day_calls=st.slider("Total number of day calls",0,165,1)
    total_day_charge=st.slider("Total charge of day calls",0.0,59.76,1.5)
    total_eve_minutes=st.slider("Total minutes of evening calls",0.0,359.3,1.5)
    total_eve_calls=st.slider("Total number of evening calls",0,170,1)
    total_eve_charge=st.slider("Total charge of evening calls",0.0,30.54,1.5)
    total_night_minutes=st.slider("Total minutes of night calls",0.0,395.0,1.5)
    total_night_calls=st.slider("Total number of night calls",0,175,1)
    total_night_charge=st.slider("Total charges of night calls",0.0,17.77,1.5)
    total_intl_minutes=st.slider("Total minutes of international calls",0.0,20.0,1.5)
    total_intl_calls=st.slider("Total number of international calls",0,20,1)
    total_intl_charge=st.slider("Total charge of international calls",0.0,5.4,0.5)
    number_customer_service_calls=st.slider("Number of calls to customer service",0,9,1)
    
    def encoding_plan(df):
        df['international_plan'].replace( to_replace='no', value=0,inplace=True )
        df['international_plan'].replace( to_replace='yes', value=1,inplace=True )
        df['voice_mail_plan'].replace( to_replace='no', value=0, inplace=True )
        df['voice_mail_plan'].replace( to_replace='yes', value=1, inplace=True )
        return df


    ok = st.button("Predict")
    columns_name = [ 'account_length', 'area_code' ,'international_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'number_customer_service_calls']

    values = [ account_length,area_code ,international_plan, voice_mail_plan,
       number_vmail_messages, total_day_minutes, total_day_calls,
       total_day_charge, total_eve_minutes, total_eve_calls,
       total_eve_charge, total_night_minutes, total_night_calls,
       total_night_charge, total_intl_minutes, total_intl_calls,
       total_intl_charge, number_customer_service_calls]

    numerical_features_name = ['total_night_charge', 'total_day_charge', 'total_day_calls', 'account_length',
                                'total_intl_charge', 'total_eve_charge', 'total_night_calls', 'total_eve_minutes',
                                 'total_intl_minutes', 'number_vmail_messages', 'total_intl_calls', 'total_night_minutes',
                                   'total_eve_calls', 'total_day_minutes', 'number_customer_service_calls']


    if ok:
        df = pd.DataFrame( data=[values] ,columns=columns_name )
                
        df[numerical_features_name] = RS.transform(df[numerical_features_name])

        temp = ohe.transform( df[['area_code']] )
        temp = pd.DataFrame( temp )
        df.drop(columns='area_code',axis=1,inplace=True)
        df = pd.concat( [df,temp],axis=1 )

        df = encoding_plan(df)
    
        predict_value = model.predict(df)
        
        if predict_value == 1:
            st.subheader("Predicted Churn: Yes ")
        else:
            st.subheader("Predicted Churn: No ")
