import ollama 
import json 

class LLMAnalyst:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name

    def generate_insights(self, job_data, predicted_salary, market_avg):
        # we will use a structured prompt to ensure story telling and visualization
        promt = f"""
        You are a Senior Data Science Career Consultant.
        Analyze this predicted salary: ${predicted_salary:,.2f}

        Job Context:
        - Job Title: {job_data['job_title']}
        - Experience Level: {job_data['experience_level']}
        - Company Size: {job_data['company_size']}
        - Market Average for this role: ${market_avg:,.2f}

        Tasks:
        1. Narrative: 3-4 sentences of 'storytelling' analysis. Explain why this specific 
           salary was predicted based on the experience and company size.
        2. Chart: Create a comparison chart called 'Salary vs Market'.
        
        You MUST return your response in JSON format with these exact keys:
        {{
            "narrative": "your storytelling text",
            "chart_title": "Salary Comparison",
            "chart_data": {{
                "labels": ["Predicted Salary", "Market Average"],
                "values": [{predicted_salary}, {market_avg}]
            }}
        }}
        """

        try:
            # we used format = json here to ensure the output is machine-readable
            response = ollama.chat(
                model=self.model_name, 
                messages=[{"role": "user", "content": promt}],
                format="json"
            )
            # parse string into a python dict
            return json.loads(response['message']['content'])
        except Exception as e:
            return {
                "narrative": f"Error generating insights: {str(e)}",
                "chart_data": None
            }
        
