
# Risk-Based Recommender System of Active and Passive Portfolios of Replica

*A machine learning-driven system for personalized portfolio recommendations, combining index replication and active anomaly-responsive strategies.*


## Project Overview

This project folder contains the **Risk-Based Recommender System for Active and Passive Portfolios of Replica**, a project that applies machine learning and risk management techniques to personalize investment portfolio recommendations. The system classifies investors into tailored profiles (such as *Accumulation* or *Income*) and suggests portfolios that match their risk appetite and financial goals. It combines both **passive** index-tracking portfolios and **active** portfolios that adjust dynamically to market anomalies and stress conditions. The objective is to leverage data-driven insights to help investors achieve better risk-adjusted returns through a mix of traditional index replication and active management strategies.

This project was developed as the final assignment for a Fintech course (Politecnico di Milano), demonstrating an end-to-end pipeline from data processing and model tuning to a deployed web application.


## Project Structure

The repository is organized into several components, each encapsulating a part of the pipeline:

- **Group13_Final_Project.ipynb**– *Main Pipeline Notebook*. This Jupyter notebook orchestrates the entire workflow.

- **Predictor/** – *Recommender System Module*. Contains helper functions and Optuna hyperparameter tuning scripts for the recommendation engine. It also contains the folder **ModelForTheWebsite/** where the calibrated models for deploying the recommendation system as a web app are saved.

- **PortfolioReplica/** – *Portfolio Replication Module*. Includes code for constructing and tuning index-tracking portfolios.

- **AnomalyDetection/** – *Market Anomaly Detection Module*. Implements algorithms to identify market anomalies and stress events.

- **Datasets/** – *Data Directory*. Includes all datasets needed for model training and evaluation.

- **WebApp/** – *Web Application Code*. This folder contains the source code of the web app that is live at https://gm46.pythonanywhere.com/. This code is provided for reference and should not be run locally, as the hosted version is already live.


## Code Execution

1. **Use a locally installed IDE for running the notebooks**. This project includes Jupyter notebooks and was developed using VSCode’s Python/Jupyter environment. While you can run the code in a Jupyter Notebook server or another IDE, some functions may not work reliably in cloud-based environments like Google Colab (due to file system or environment differences). For the smoothest experience, open the project in Visual Studio Code with the Jupyter extension.

2. **Install the requirements** by running cell 6 you ensure to install all the needed packages.


## Usage

1. Open `Group13_Final_Project.ipynb`.

2. Run the notebook to execute data loading, anomaly detection, model training, and portfolio recommendation.

3. Outputs include anomaly detection results, model metrics, and portfolio compositions.

4. The Jupyter file is built to replicate one portfolio per time. By default, the code replicates the Equity Portfolio aimed at replicating the MXWO Index. In order to replicate the other Portfolios, go to Section '0. Selection of the portfolio', choose the desired portfolio by selecting the corresponding button under cell 6 of the Jupyter file and then run the code again from cell 8 to the end.

5. To ease the running of the main Jupyter Notebook, the Optuna number of trials was put to 0, since the hyperparameters calibration has already been performed. If you want to further proceed with the hyperparameters tuning, remember to change the variables n_trials at cell 26 (to recalibrate the index replication) and optuna_n_trials at cell 53 (to recalibrate the anomaly detection models).

6. In Section '5. Replica and Anomaly Detection Combination - Alternative Portfolios: Active Risk Management with Anomaly Detection' the index replica and the anomaly detection system are combined based on the best last calibrated results. If you want to combine other models, you can go to cell 91 and modify:
	
- the CSV file from which the anomaly detection are taken (the CSV files names can be found in the folder /AnomalyDetection/CSVCalibbratedResults and they are updated everytime the code runs).

- the variables named base_positions and model_results; in particular, modify them as follows:
	
	base_positions = positions_history_df_EN.copy()	-> to replicate the index via Elastic Net model	
	model_results = results_EN.copy()	
			
		
	base_positions = positions_history_df_KF.copy() -> to replicate the index via Kalman Filter model
	model_results = result_KF.copy()	 

	positions_history_df_KFE.copy() 		-> to replicate the index via Ensemble Kalman Filter model
	model_results = result_KFE.copy()


## Website Deployment

One of the outcomes of this project is a web application that allows users to input their data and receive portfolio recommendations in real time. The code for this web app is contained in the ModelForTheWebsite directory. It uses a lightweight web framework (Flask) along with the trained model to serve predictions.

Live Demo: A demo of the web application is hosted at gm46.pythonanywhere.com. On this site (branded “The Big Levy Investments” as a mock investment platform), users can fill out a short questionnaire (e.g. age, income, risk preference) and the system will recommend one or more investment portfolios (either passive ETF portfolios or active “Alternative” portfolios) tailored to their profile. This demo showcases the recommender system in action.

Note: The web application and the recommendation results are for educational purposes to demonstrate the model’s capabilities. They are not intended as real investment advice.


## Acknowledgments

This project was created as a final assignment for the 2024/2025 Fintech course at Politecnico di Milano by Group 13, composed by Manfredi Giacomo, Pescotto Leonardo, Tarditi Andrea, Toia Nicolo' and Torba Matteo.

