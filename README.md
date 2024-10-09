# Bank Customer Churn Prediction Model

## Project Overview
This project focuses on developing a **bank customer churn prediction** model using **Python**. The aim is to identify factors influencing customers' decisions to leave the bank and predict future churn. The solution is built with **Random Forest** and **XGBoost** algorithms, which were chosen for their efficiency in handling structured data and providing high predictive accuracy. The project involves data processing, model implementation, and deployment using web technologies for easy integration into production environments.

## Features
- **Churn Prediction**: Predict whether a customer is likely to churn based on historical data.
- **Machine Learning Models**: Includes models built with:
  - **Random Forest**
  - **XGBoost**
- **Data Processing**: Uses **Pandas** for data manipulation and cleaning.
- **Deployment**: Integrated with **FastAPI** or **Streamlit** to provide a user-friendly web interface for prediction.

## Technology Stack
- **Python**: Core programming language for data analysis and model building.
- **Scikit-learn**: Machine learning library used to implement Random Forest and model evaluation.
- **XGBoost**: An advanced implementation of gradient boosting for enhanced prediction accuracy.
- **Pandas**: Library for data manipulation and preprocessing.
- **FastAPI** / **Streamlit**: Frameworks for deploying the model as a web application.

## Project Setup

### Prerequisites
- Python 3.x
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model Performance
- **Random Forest**: Achieved high predictive accuracy with robust performance on a wide range of customer datasets.
- **XGBoost**: Performed exceptionally well with faster training times and improved accuracy.
- **Evaluation Metrics**: Both models were evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Future Improvements
- Experiment with additional machine learning models to further improve prediction accuracy.
- Optimize hyperparameters for both models using techniques such as grid search or Bayesian optimization.
- Enhance the user interface for better integration into production environments.

## Acknowledgments
Special thanks to the open-source libraries **Scikit-learn**, **XGBoost**, and the web frameworks **FastAPI** and **Streamlit** for enabling efficient model building and deployment.
