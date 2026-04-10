import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SupabaseService:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
            
        if not url or not key:
            raise ValueError("❌ Error: SUPABASE_URL and SUPABASE_KEY not found in .env")
                
        self.supabase: Client = create_client(url, key)
    
    def save_prediction(self, job_data, predicted_salary, llm_report):
        """
        Requirement 6: Persist prediction and LLM results to Supabase.
        """
        try:
            # Construct the payload matching our SQL schema
            payload = {
                "job_title": job_data["job_title"],
                "experience_level": job_data["experience_level"],
                "employment_type": job_data["employment_type"],
                "company_size": job_data["company_size"],
                "predicted_salary": predicted_salary,
                "llm_report": llm_report # This is the dict from LLMAnalyst
            }
            
            # Insert into Supabase
            result = self.supabase.table("salary_predictions").insert(payload).execute()
            print("✅ Data successfully persisted to Supabase.")
            return result
        except Exception as e:
            print(f"❌ Supabase Error: {e}")
            return None