import streamlit as st
import pandas as pd

user_input = st.text_input("Enter your data here:")
df = pd.DataFrame([user_input.split(" ")], columns=["Column 1", "Column 2"])
df.to_csv("example.csv", index=False)

if st.button("Download CSV"):
    with open("example.csv", "r") as f:
        st.download_button(
            label="Download CSV",
            data=f.read(),
            file_name="example.csv",
            mime="text/csv",
        )
