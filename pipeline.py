import requests
import pandas as pd
import os
import joblib
from src.llm_analyst import LLMAnalyst
from src.database import SupabaseService

# configuration
API_URL = "http://localhost:8000/predict"
DATA_PATH = "data/processed_salaries.csv"
ENCODER_PATH = "models/encoders.joblib"

def get_market_context(job_title_str):
    #Provides the LLM with real data to improve 'Storytelling' accuracy.

    try:
        df = pd.read_csv(DATA_PATH)
        encoders = joblib.load(ENCODER_PATH)

        # encode the job title string to its numeric ID
        le = encoders['job_title']
        encoded_title = le.transform([job_title_str])[0]

        # Find the average salary in the dataset for this specific job title
        avg_salary = df[df['job_title'] == encoded_title]['salary_in_usd'].mean()

        # fallback if this title has no data (unlikely but more robust)
        if pd.isna(avg_salary):
            return df['salary_in_usd'].mean()
        
        return avg_salary
    except Exception as e:
        print(f"⚠️ Warning: Could not calculate market context ({e}). Using global average.")
        return 112297.87 # global average from df.describe() output

    

def run_pipeline(scenario):
    # script that calls the API, handeling errors and coordinates the flow from API -> LLM
    print(f"\n🚀 Processing Scenario: {scenario['job_title']} ({scenario['experience_level']})")
    
    # 1. Call the FastAPI
    try:
        # We pass parameters as a query string because it's a GET API
        response = requests.get(API_URL, params=scenario, timeout=10)
        response.raise_for_status() # Raises error for 4xx or 5xx status codes
        
        predicted_salary = response.json()['predicted_salary_usd']
        print(f"💰 API Prediction: ${predicted_salary:,.2f}")

        # 2. Get real context for the LLM
        market_avg = get_market_context(scenario['job_title']) 
        print(f"📊 Market Baseline for '{scenario['job_title']}': ${market_avg:,.2f}")

        # 3. Requirement 5: Call LLM for Narrative + Chart Data
        print("🤖 LLM analyzing 'Storytelling' and generating chart data...")
        analyst = LLMAnalyst()
        report = analyst.generate_insights(scenario, predicted_salary, market_avg)
        
        print(f"\n--- NARRATIVE ---\n{report.get('narrative', 'No narrative generated.')}")
        
        if report.get('chart_data'):
            print(f"\n--- CHART DATA GENERATED ---")
            print(f"Labels: {report['chart_data']['labels']}")
            print(f"Values: {report['chart_data']['values']}")

        # IMPORTANT: This 'report' dictionary is exactly what we will 
        db = SupabaseService()
        db.save_prediction(scenario, predicted_salary, report)

        print(f"💰 Saved Prediction: ${predicted_salary:,.2f}")
        print(f"📄 Saved Narrative: {report.get('narrative')[:50]}...")

    except requests.exceptions.RequestException as e:
        print(f"❌ API Connection Error From Pipeline: {e}")

if __name__ == "__main__":
    # Requirement 4: "Cover the input space"
    # We define a few different scenarios to test the robustness of the system
    test_scenarios = [
        {"work_year": 2026, "experience_level": "SE", "employment_type": "FT", "job_title": "Data Scientist", "employee_residence": "US", "remote_ratio": 100, "company_location": "US", "company_size": "L"},
        {"work_year": 2026, "experience_level": "EN", "employment_type": "FT", "job_title": "Data Analyst", "employee_residence": "GB", "remote_ratio": 0, "company_location": "GB", "company_size": "S"}
    ]

    for scenario in test_scenarios:
        run_pipeline(scenario)