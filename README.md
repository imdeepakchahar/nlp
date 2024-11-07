# Crime Data Prediction App

This web application predicts the **category** and **sub-category** of a crime based on the provided crime description. The application uses machine learning models trained on crime data, processes input text, and returns the corresponding predictions. It is built using **Flask**, **Scikit-learn**, and **LangChain**.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Categories](#predicting-categories)
- [URLs](#urls)
  - [Train Endpoint](#train-endpoint)
  - [Predict Endpoint](#predict-endpoint)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Crime Data Prediction App uses a machine learning model to classify crime descriptions into categories and sub-categories. It supports Hinglish (a mix of Hindi and English) text by utilizing LangChain for preprocessing. The app allows users to enter crime descriptions and receive predictions for their category and sub-category. It also includes a training feature to allow users to retrain the models based on updated data.

## Features

- **Text Preprocessing**: Cleans and prepares input data, including translation to English if necessary.
- **Model Training**: Trains a model to predict the category and sub-category of crime descriptions.
- **Prediction**: Predicts the crime category and sub-category based on input descriptions.
- **Flask Web Interface**: Provides a simple UI for users to interact with the application.
- **Hinglish Support**: Translates Hinglish text into clean English for better prediction accuracy(future intigration).

## Technologies Used

- **Flask**: Web framework for building the application.
- **Scikit-learn**: Machine learning library used for building the predictive models.
- **LangChain**: For processing Hinglish (mixed Hindi and English) text and translating it to English.
- **Pandas**: Used for data manipulation and handling.
- **Joblib**: For saving and loading machine learning models.

## Installation

Follow these steps to set up the application:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crime-data-prediction.git
   cd crime-data-prediction
