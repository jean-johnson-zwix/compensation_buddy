import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import model_layer

st.set_page_config(
    page_title="Compensation Buddy",
    layout="centered"
)

st.title("Compensation Buddy 😊")
st.markdown("I can help you figure out you salary based on role, location, and company type")

#---------- User Input Form ----------------
st.subheader("Tell me about you!")

col1, col2 = st.columns(2)

with col1:
    role = st.selectbox("Role Category", [
        'Software Engineering', 'Data Engineering', 'Data Science',
        'ML/AI', 'Analytics', 'Database', 'QA/Testing', 
        'Infrastructure', 'Management'
    ])

    seniority = st.selectbox("Seniority Level", [
        'Entry', 'Mid', 'Senior', 'Staff', 'Principal', 
        'Manager', 'Leadership'
    ])

    industry = st.selectbox("Industry", [
        'Big Tech', 'Finance', 'Consulting', 'Other'
    ])

with col2:
    wage_level = st.selectbox("Wage Level", ['I', 'II', 'III', 'IV'])

    metro_tier = st.selectbox("Metro Tier", ['Tier1', 'Tier2', 'Tier3'])

    state = st.selectbox("Work State", [
        'CA', 'WA', 'NY', 'TX', 'IL', 'MA', 'GA', 'NC', 'NJ', 'VA',
        'FL', 'CO', 'MN', 'MI', 'OH', 'PA', 'AZ', 'OR', 'MD', 'CT',
        'UT', 'IN', 'TN', 'MO', 'WI', 'KY', 'AL', 'OK', 'IA', 'KS',
        'NV', 'AR', 'SC', 'NE', 'ID', 'NH', 'ME', 'RI', 'MT', 'DE',
        'SD', 'ND', 'WV', 'AK', 'HI', 'DC', 'PR', 'GU', 'VT', 'WY',
        'MS', 'NM', 'LA', 'Other'
    ])

    is_top_tier = st.checkbox("Top Tier Company (Google, Microsoft, Apple, Meta, Amazon etc.)")

#------------ Predict Button------------
st.divider()

if st.button("Predict my salary", type="primary"):
    
    X = model_layer.build_input(
        role, seniority, industry, wage_level,
        metro_tier, state, is_top_tier
    )
    
    salary = model_layer.predict_salary(X)
    
    # Display result
    st.success(f"### Your predicted Annual Salary: ${salary:,.0f}")
    
    # Add context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted", f"${salary:,.0f}")
    with col2:
        st.metric("Low Estimate", f"${salary * 0.85:,.0f}")
    with col3:
        st.metric("High Estimate", f"${salary * 1.15:,.0f}")
    