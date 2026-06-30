import os
import joblib
import json
import pandas as pd
from flask import Flask, render_template, request
from Predictor import Predictor

# Initialize Flask app with the correct template and static folder paths
app = Flask(__name__,
            template_folder='/home/gm46/templates',  # Point to the correct templates folder
            static_folder='/home/gm46/mysite/static')  # Point to the correct static folder for images

# Load model and tau_dict once at startup
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'joint_model.joblib')
tau_path = os.path.join(base_path, 'tau_dict.json')

# Initialize Predictor and load model and tau_dict
p = Predictor(model=None)
p.joint_model = joblib.load(model_path)
with open(tau_path, 'r') as f:
    p.tau_dict = json.load(f)

# Portfolio dictionary with risk values
ptfs_dict = {
    'ETF_bond': 1/7,
    'ETF_equity': 3/7,
    'ETF_life': 2/7,
    'active_bond': 2/7,
    'active_equity': 4/7,
    'active_life': 3/7,
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if all form data is provided
            age = request.form.get('Age')
            gender = request.form.get('Gender')
            family_members = request.form.get('FamilyMembers')
            financial_education = request.form.get('FinancialEducation')
            risk_propensity = request.form.get('RiskPropensity')
            wealth = request.form.get('Wealth')
            income = request.form.get('Income')

            # Check if any required field is empty
            if not all([age, gender, family_members, financial_education, risk_propensity, wealth, income]):
                raise ValueError("Please fill out all the fields.")

            # Convert form data to the appropriate types
            age = float(age)
            gender = int(gender)
            family_members = int(family_members)
            financial_education = float(financial_education)
            risk_propensity = float(risk_propensity)
            wealth = float(wealth)
            income = float(income)

            # Prepare dataframe for prediction
            new_client = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "FamilyMembers": [family_members],
                "FinancialEducation": [financial_education],
                "RiskPropensity": [risk_propensity],
                "Wealth": [wealth],
                "Income": [income],
            })



            # Predict using your Predictor method
            (acc_pred, acc_prob), (inc_pred, inc_prob) = p.pred_new_client_joint(new_client)
            if inc_prob > 0.4:
                inc_pred = 1
            if acc_prob > 0.4:
                acc_pred = 1
            # Suggest portfolios based on predictions
            ptf_pred = {}

            if acc_pred == 1:  # If accumulation prediction is 1
                if inc_pred == 1:  # If income prediction is 1
                    for name, ptf_risk in ptfs_dict.items():
                        if ptf_risk <= risk_propensity:
                            ptf_pred[name] = ptf_risk
                    sorted_ptf = dict(sorted(ptf_pred.items(), key=lambda x: x[1], reverse=True))

                    top_portfolios = list(sorted_ptf.items()) if ptf_pred else []

                else:  # Acc 1, Inc 0
                    for name, ptf_risk in ptfs_dict.items():
                        if name in ['ETF_life', 'active_life', 'ETF_bond', 'ETF_equity'] and ptf_risk <= risk_propensity:
                            ptf_pred[name] = ptf_risk
                    sorted_ptf = dict(sorted(ptf_pred.items(), key=lambda x: x[1], reverse=True))

                    top_portfolios = list(sorted_ptf.items()) if ptf_pred else []

            else:  # If accumulation prediction is 0
                if inc_pred == 1:  # Acc 0, Inc 1
                    for name, ptf_risk in ptfs_dict.items():
                        if name in ['active_equity', 'active_bond', 'ETF_bond', 'ETF_equity'] and ptf_risk <= risk_propensity:
                            ptf_pred[name] = ptf_risk
                    sorted_ptf = dict(sorted(ptf_pred.items(), key=lambda x: x[1], reverse=True))

                    top_portfolios = list(sorted_ptf.items()) if ptf_pred else []

                else:  # Acc 0, Inc 0
                    top_portfolios = [('ETF_bond', ptfs_dict['ETF_bond'])]  # Default portfolio if no match

            if acc_pred == 1:
                if inc_pred == 1:
                    message = "Your profile falls into BOTH CATEGORIES. Below, you’ll find the products that best match your investment style and risk tolerance. You can always view our full range of offerings on the homepage of our website. In any case, we encourage you to do your own research and study before committing to any investment."
                else:
                    message = "Your profile falls into the ACCUMULATION category. Below, you’ll find the products that best match your investment style and risk tolerance. You can always view our full range of offerings on the homepage of our website. In any case, we encourage you to do your own research and study before committing to any investment."
            else:
                if inc_pred == 1:
                    message = "Your profile falls into the INCOME category. Below, you’ll find the products that best match your investment style and risk tolerance. You can always view our full range of offerings on the homepage of our website. In any case, we encourage you to do your own research and study before committing to any investment."
                else:
                    message = """Oops! No Suitable Investment Approach Detected.
                            Based on the information you provided, our machine-learning model cannot confidently recommend either
                            the Accumulation or Income approach at this time. This result often occurs when current income, savings,
                            or overall net worth may be insufficient to support a diversified investment plan without placing undue strain
                            on day-to-day finances.
                            What to do next
                            • Consider building an emergency fund and reducing any high-cost debt before you begin investing.
                            • Invest in yourself first: deepen your understanding of financial markets, risk, and long-term planning
                            through reputable courses, books, and trusted educational resources like the Quantitative Finance
                            course from Politecnico di Milano.
                            • Revisit your finances periodically; as your income or savings grow, one of our tailored investment paths
                            may become appropriate.
                            In the meantime, here’s a snapshot of our lowest-risk portfolio for your review. We still encourage you
                            to keep reading, stay informed, and follow the markets before committing any capital."""

            if risk_propensity < 1/7:
                message2 = """Oops! We couldn’t generate a recommendation this time.
                    It appears that your current risk profile doesn’t align with any of our existing portfolios. We suggest
                    exploring lower-risk options for now and taking a bit of time to learn more about investing. A deeper
                    understanding of how financial markets work can make risk feel less daunting and help you refine your
                    investment preferences.
                    In the meantime, here’s a snapshot of our lowest-risk portfolio for your review. We still encourage you
                    to keep reading, stay informed, and follow the markets before committing any capital."""
                top_portfolios = [('ETF_bond', ptfs_dict['ETF_bond'])]  # Default portfolio if no match
            else:
                message2 =""
            # Format probabilities nicely (optional)
            acc_prob_str = f"{acc_prob:.2f}"
            inc_prob_str = f"{inc_prob:.2f}"


            # Portfolio images paths (relative to the static folder)
            portfolio_images = {
                'ETF_bond': 'images/ETF_Bond.png',
                'ETF_equity': 'images/ETF_Equity.png',
                'ETF_life': 'images/ETF_Life.png',
                'active_bond': 'images/Active_Bond.png',
                'active_equity': 'images/Active_Equity.png',
                'active_life': 'images/Active_Life.png',
            }

            # Render the prediction results along with the portfolio suggestions
            return render_template('predict.html',
                                   prediction_acc=acc_pred,
                                   prob_acc=acc_prob_str,
                                   prediction_inc=inc_pred,
                                   prob_inc=inc_prob_str,
                                   message=message,
                                   message2=message2,
                                   input_data=request.form,
                                   top_portfolios=top_portfolios,
                                   portfolio_images=portfolio_images)  # top_portfolios is already a list

        except Exception as e:
            return render_template('predict.html', error=str(e))

    return render_template('predict.html')
@app.route('/team')
def team():
    return render_template('team.html')
if __name__ == '__main__':
    app.run(debug=True)
