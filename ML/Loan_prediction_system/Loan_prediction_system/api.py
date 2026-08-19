from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import pandas as pd 
from loan_model import LoanApprovalApp

app = FastAPI(
    title = "Loan Approval Api", 
    description= "two stage ML model to predict Loan approval and amount"
)

model = LoanApprovalApp()

class ApplicantData(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


    @app.post("/predict")
    def predict_loan(data : ApplicantData):
        try:
            applicant_df = pd.DataFrame([data.model_dump()])
            result = model.two_stage_predict(applicant_df)

            return {
                "status" : "success" , 
                "data": result
            }
        except Exception as e:
            raise HTTPException ( status_code = 500 , detail = str(e))