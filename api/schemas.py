from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    work_year: int = Field(..., example=2026)
    experience_level: str = Field(..., description="EN, MI, SE, EX")
    employment_type: str = Field(..., description="FT, PT, CT, FL")
    job_title: str
    employee_residence: str
    remote_ratio: int = Field(..., ge=0, le=100)
    company_location: str
    company_size: str = Field(..., description="S, M, L")