import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

page_title = "Daibetes_Prediction and checkup"

st.set_page_config(page_title = page_title)

df = pd.read_csv("diabetes_1.csv")


# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
name = st.sidebar.text_input('Enter the name')
st.write(df.head())
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
    pregnancies = st.sidebar.text_input('Pregnancies')
    glucose = st.sidebar.text_input('Glucose')
    bp = st.sidebar.text_input('Blood Pressure')
    skinthickness = st.sidebar.text_input('Skin Thickness')
    insulin = st.sidebar.text_input('Insulin')
    bmi = st.sidebar.text_input('BMI')
    dpf = st.sidebar.text_input('Diabetes Pedigree Function')
    age = st.sidebar.text_input('Age')

    user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

#submitted= st.sidebar.form_submit_button("Save Data")
#if submitted:
#   st.success("Data Saved")


# PATIENT DATA
user_data_with_name = user_report()
#user_data_with_name = user_report()
user_data_with_name.insert(0,"name",name,True)
#df.insert(2, "Age", [21, 23, 24, 21], True)
st.subheader('Patient Data')
st.write(user_data_with_name)
user_data_witout_name = user_data_with_name.drop('name',axis=1)

# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data_witout_name)



# VISUALISATIONS 
st.title('Visualised Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
    color = 'blue'
else:
    color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
    
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
