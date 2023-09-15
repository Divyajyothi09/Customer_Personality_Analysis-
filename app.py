import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler


kmeans = pickle.load(open('C:/Users/anvesh/Desktop/customer_segmentation/kmeansmodel.pkl', 'rb'))



def pred(input_data):
    st.title("KMeans clustering app")
    data1=np.array(input_data)
    data2=data1.reshape(1,-1)
    scaler=StandardScaler()
    data3=scaler.fit_transform(data2)
    prediction=kmeans.predict(data3)
    print(prediction)
    
    
    
    print(prediction)
    
    if (prediction == 0):
        return 'cluster 0'
   
    elif (prediction == 1):
        return'cluster 1'
    
    elif (prediction == 2):
        return 'cluster 2'
    
    elif (prediction == 3):
        return 'cluster 3'
        
   
        
    
def main():
    st.title("KMeans modelling")
    
    
    Income=st.number_input("Enter Income")
    Recency=st.number_input("Enter Recency")
    NumDealsPurchases=st.number_input("Enter NumDealsPurchases")
    NumWebPurchases=st.number_input("Enter NumWebPurchases")
    NumCatalogPurchases=st.number_input("Enter NumCatalogPurchases")
    NumStorePurchases=st.number_input("NumStorePurchases")
    NumWebVisitsMonth=st.number_input("NumWebVisitsMonth")
    Complain=st.number_input("Complain")
    Response=st.number_input("Response")
    Age=st.number_input("Age")
    Expenses=st.number_input("Expenses")
    Children=st.number_input("Children")
    Campaign=st.number_input("Campaign")
    Education_Postgraduate=st.number_input("Education_Postgraduate")
    Education_Undergraduate=st.number_input("Education_Undergraduate")
    Marital_Status_Partner=st.number_input("Marital_Status_Partner")
    
    if st.button("clustering result"):
        result=pred([Income,Recency,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,Complain,Response,Age,Expenses,Children,Campaign,Education_Postgraduate,Education_Undergraduate,Marital_Status_Partner])
        st.success(result)


if __name__ == "__main__":
    main()
   