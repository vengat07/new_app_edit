#import streamlit as st

#import pandas as pd

#st.title("Excel Update App")

#df = pd.read_csv("data_names.csv") 

#st.header("Existing File")

#st.write(df)

#st.sidebar.header("Options")

#options_form = st.sidebar.form("options_form")

#user_Name = options_form.text_input("Name") 

#user_Pregnancies = options_form.text_input("Pregnancies") 

#user_Glucose = options_form.text_input("Glucose")

#user_BloodPressure = options_form.text_input("BloodPressure")

#user_SkinThickness = options_form.text_input("SkinThickness")

#user_Insulin = options_form.text_input("Insulin")

#user_BMI = options_form.text_input("BMI")

#user_DiabetesPedigreeFunction = options_form.text_input("DiabetesPedigreeFunction")

#user_Age = options_form.text_input("Age")

#add_data =  options_form.form_submit_button()

#if add_data:
 # new_data= {"Name": user_Name, "Pregnancies": int(user_Pregnancies), "Glucose": int(user_Glucose), "BloodPressure": int(user_BloodPressure), "SkinThickness": int(user_SkinThickness), "Insulin": int(user_Insulin), "BMI": int(user_BMI), "DiabetesPedigreeFunction": float(user_DiabetesPedigreeFunction), "Age": int(uses_Age)} 

  #df=df.append(new_data, ignore_index=True)

  #df.to_csv("data_names.csv", index=False)
  
import streamlit as st
import pandas as pd

st.title("Excel Update App")
df = pd.read_csv("data__names.csv") 
st.header("Existing File")
st.write(df)

st.sidebar.header("Options")
options_form = st.sidebar.form("options_form")
user_name =options_form.text_input("Name") 
user_age =options_form.text_input("Age") 
add_data =options_form.form_submit_button()
if add_data:
    new_data= {"name": user_name, "age": int(user_age)} 
    df=df.append(new_data, ignore_index=True)
    df.to_csv("data__names.csv", index=False)
