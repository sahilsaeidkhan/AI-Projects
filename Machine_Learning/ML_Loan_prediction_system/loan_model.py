import joblib
import os
import pandas as pd


class LoanApprovalApp:

    def __init__(
        self,
        classifier="stage_1_rf_classifier_pipeline.pkl",
        regressor="stage_2_rf_regression_pipeline.pkl"
    ):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(
            BASE_DIR,
            "models"
        )

        self.clf = joblib.load(
            os.path.join(model_path, classifier)
        )

        self.reg = joblib.load(
            os.path.join(model_path, regressor)
        )

    def get_user_input(self):

        print("--- Enter Applicant Details ---\n")

        data = {
            "no_of_dependents": 3,
            "education": "Graduate",
            "self_employed": "Yes",
            "income_annum": 8300000,
            "loan_amount": 3001400000,
            "loan_term": 60,
            "cibil_score": 900,
            "residential_assets_value": 1000000,
            "commercial_assets_value": 1600000,
            "luxury_assets_value": 17200000,
            "bank_asset_value": 6100000
        }

        return pd.DataFrame([data])

    def two_stage_predict(self, applicant_df):

        out = {}

        # Stage 1: Predict loan approval
        approve = self.clf.predict(applicant_df)[0]

        print("Approve:", approve)

        out["approve"] = int(approve)

        # Stage 2: Predict loan amount only if approved
        if approve == 1:

            applicant_df_reg = applicant_df.copy()

            applicant_df_reg["loan_status"] = "approved"

            pred = self.reg.predict(applicant_df_reg)[0]

            out["regression_prediction"] = float(pred)

        else:

            out["regression_prediction"] = None

            print("Not Approved")

        return out

    def run(self):

        applicant_df = self.get_user_input()

        result = self.two_stage_predict(applicant_df)

        print("\n------ RESULT ------")

        if result["approve"] == 1:

            print("Loan Status : ✅ APPROVED")

            print(
                f"Predicted Loan Amount / Value : "
                f"{result['regression_prediction']:.2f}"
            )

        else:

            print("Loan Status : ❌ REJECTED")

        print("---------------------")

        return result


if __name__ == "__main__":

    loan_approval_application = LoanApprovalApp()

    loan_approval_application.run()