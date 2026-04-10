import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.database import SupabaseService
from src.llm_analyst import LLMAnalyst
from pipeline import get_market_context

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Salary AI Insights", page_icon="📈", layout="wide")

# Everforest Hard Dark Theme Injection
st.markdown("""
    <style>
    .stApp {
        background-color: #2b3339;
        color: #d3c6aa;
    }
    h1, h2, h3 {
        color: #a7c080 !important;
    }
    .stMetric {
        background-color: #323c41;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #414b50;
    }
    div[data-testid="stExpander"] {
        background-color: #323c41;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🌲 Everforest Salary Analyst")
    
    # Create Tabs for a modern feel
    tab_view, tab_new = st.tabs(["📈 Past Analytics", "➕ New Prediction"])

    db = SupabaseService()

    # --- TAB 1: VIEW PAST PREDICTIONS ---
    with tab_view:
        data = db.get_predictions()
        if not data:
            st.info("No predictions yet. Head over to the 'New Prediction' tab!")
        else:
            col_list, col_detail = st.columns([1, 2])
            
            with col_list:
                options = [f"{row['job_title']} ({row['experience_level']}) - {row['created_at'][5:16]}" for row in data]
                selected_index = st.selectbox("Select from History", range(len(options)), format_func=lambda x: options[x])
                selected_row = data[selected_index]
            
            with col_detail:
                report = selected_row['llm_report']
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted Salary", f"${selected_row['predicted_salary']:,.2f}")
                    st.write(f"**Company:** {selected_row['company_size']} | **Type:** {selected_row['employment_type']}")
                
                st.subheader("🤖 Analysis")
                st.info(report.get('narrative'))

                if report.get('chart_data'):
                    chart = report['chart_data']
                    fig = go.Figure(go.Bar(
                        x=chart['labels'], y=chart['values'],
                        marker_color=['#a7c080', '#dbbc7f'],
                        text=[f"${v:,.0f}" for v in chart['values']], textposition='auto'
                    ))
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                    font=dict(color='#d3c6aa'), margin=dict(t=10, b=10))
                    # FIXED: use on_container_width='stretch' for modern Streamlit
                    st.plotly_chart(fig, width='stretch')

    # --- TAB 2: CREATE NEW PREDICTION ---
    with tab_new:
        st.subheader("Run Live Pipeline")
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                title = st.text_input("Job Title", value="Data Scientist")
                year = st.number_input("Year", value=2026)
            with c2:
                exp = st.selectbox("Experience", ["EN", "MI", "SE", "EX"])
                etype = st.selectbox("Type", ["FT", "PT", "CT", "FL"])
            with c3:
                csize = st.selectbox("Company Size", ["S", "M", "L"])
                remote = st.slider("Remote %", 0, 100, 100)
            
            # Hardcoded locations for the demo
            residence = "US"
            location = "US"

            submit = st.form_submit_button("🚀 Run Prediction & AI Analysis")

        if submit:
            with st.spinner("Talking to API and Ollama..."):
                try:
                    # 1. API Call
                    scenario = {
                        "work_year": year, "experience_level": exp, "employment_type": etype,
                        "job_title": title, "employee_residence": residence,
                        "remote_ratio": remote, "company_location": location, "company_size": csize
                    }
                    resp = requests.get(API_URL, params=scenario)
                    pred = resp.json()['predicted_salary_usd']

                    # 2. LLM Analysis
                    market_avg = get_market_context(title)
                    report = LLMAnalyst().generate_insights(scenario, pred, market_avg)

                    # 3. Save to Supabase
                    db.save_prediction(scenario, pred, report)
                    
                    st.success(f"Success! Predicted ${pred:,.2f}. Switch to 'Past Analytics' to see the full report.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}. Is your FastAPI and Ollama running?")

if __name__ == "__main__":
    main()