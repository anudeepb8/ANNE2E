#Importing required Libraries

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load trained model

model=tf.keras.models.load_model('model.h5')

#Load StandardScaler,LabelEncoder,OneHotEncoder
with open('Standardscaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('Onehotencoder.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

#Streamlit App
st.title('Customer Churn Prediction')
#User Input
RowNumber =st.slider('RowNumber',1,10)
CreditScore=st.number_input('CreditScore')
Gender=st.selectbox('Gender',label_encoder_gender.classes_)
Age=st.slider('Age',18,92)
Tenure=st.slider('Tenure',0,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('HasCrCard',[0,1])
IsActiveMember=st.selectbox('IsActiveMember',[0,1])
EstimatedSalary=st.number_input('EstimatedSalary')
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])

input_data=pd.DataFrame({
    'RowNumber':[RowNumber],
    'CreditScore' : [CreditScore],
    'Gender'      :[label_encoder_gender.transform([Gender])[0]],
    'Age'         :[Age],
    'Tenure'      :[Tenure],
    'Balance'     :[Balance],
    'NumOfProducts' :[NumOfProducts],
    'HasCrCard' :[HasCrCard],
    'IsActiveMember' :[IsActiveMember],
    'EstimatedSalary':[EstimatedSalary], 
})

#Onehot Encode Geography
geo_encoded=label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))
#Concatnation with one hot encoded data
input_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
#Scaling the Input data
input_scaled=scaler.transform(input_df)
#Predict Churn
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]
if prediction_proba>0.5:
    st.write('churn')
else:
    st.write('no churn')