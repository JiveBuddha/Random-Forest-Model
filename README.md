# Titanic Survival Prediction: Machine Learning Model

## Project Overview

In this project, we build a **machine learning model** to predict whether a passenger survived or not on the Titanic. This project uses the **Titanic dataset**, a popular dataset used for binary classification tasks. The project demonstrates the process of data preprocessing, building a machine learning model, and evaluating its performance.

The model we use here is a **Random Forest Classifier**, which is an ensemble model known for its robustness and accuracy.

Key steps in this project:
1. **Data Preprocessing**: Cleaning and preparing the data for model training.
2. **Model Building**: Using a Random Forest classifier to predict survival.
3. **Model Evaluation**: Assessing the performance of the model using metrics such as accuracy, confusion matrix, and classification report.
4. **Model Saving**: Saving the trained model for future use.

## Files in This Repository

- **machine_learning_model.py**: This Python script performs data preprocessing, builds a machine learning model, evaluates it, and saves the model.
- **titanic_model.pkl**: The trained machine learning model (Random Forest) saved as a `.pkl` file using `joblib`.
- **README.md**: This file provides an overview of the project.

## How to Run the Project

1. **Install Required Libraries**:
   Ensure you have the following libraries installed:
   ```bash
   pip install pandas scikit-learn joblib