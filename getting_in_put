import streamlit as st
import csv

input_text = st.text_input("Enter your text here")

if st.button("Save"):
    with open("input.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([input_text])
