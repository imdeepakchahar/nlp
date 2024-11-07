# Fraud Category Prediction with Crime Data

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

1. Clone the repository:
    ```bash
    git clone https://github.com/imdeepakchahar/nlp.git
    cd nlp
    ```

2. Create and activate a virtual environment(IF REQUIRED):
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install required packages:
    ```bash
    pip install MENTIONED_LIBRARIES_IN_main.py
    ```

4. Run the app:
    ```bash
    python main.py
    ```

## Usage

### Training the Model
1. Access the training endpoint to retrain models with updated data.
    ```plaintext
    http://127.0.0.1:5000/train
    ```

### Predicting Categories and Subcategories
1. Use the prediction endpoint to classify input descriptions.
    ```plaintext
    http://127.0.0.1:5000/predict
    ```

Enter a description in the form provided, and view the predicted category and subcategory.

## URLs
- **Train Model**: `/train`
- **Predict Crime Category**: `/predict`

## Contributing
Please refer to the GitHub repository for contribution guidelines.

## License
MIT License
