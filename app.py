import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Walmart Sales Analysis – Efficient Market Hypothesis")

# Load dataset
df = pd.read_csv("Walmart_Sales.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Plot Weekly Sales trend
st.subheader("Weekly Sales Trend")
fig, ax = plt.subplots(figsize=(10,4))
df.groupby('Date')['Weekly_Sales'].mean().plot(ax=ax, color='violet')
st.pyplot(fig)

# Show Average Sales per Store
st.subheader("Average Sales per Store")
store_avg = df.groupby('Store')['Weekly_Sales'].mean()
st.bar_chart(store_avg)

st.success("Analysis Complete! You can explore more features locally using Tkinter version.")
