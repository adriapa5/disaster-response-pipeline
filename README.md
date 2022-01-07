# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [File Structure](#file_structure)
	3. [Installing](#installation)
    4. [Instructions](#instructions)
	5. [Additional Material](#material)
    6. [It's Alive](#production)
3. [License](#license)
4. [Acknowledgement](#acknowledgement)


<a name="description"></a>
## Description
The aim of this project is to provide a web application to categorize messages from real-life disaster events. This is built on top of a Natural Language Processing (NLP) trained model. The dataset, used for training the model, contains pre-labelled tweet and messages from real-life disaster events. 
This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.

There are three important parts in the project:
1. Build an ETL pipeline to extract data from provided sources, clean the data and save it in a SQLite database.
2. Build a NLP model using Machine Learning pipelines to train and optimize the categorization of text messages.
3. Expose the model via web app in order to classify messages in real time.

<a name="getting_started"></a>
## Getting Started
<a name="dependencies"></a>
### Dependencies
* Python 3.7+
* SQLite database libraries: SQLalchemy
* Machine Learning libraries: Pandas, Numpy, Scikit-Learn
* Natural Language Processing libraries: NLTK
* Web App and Data Visualization libraries: Flask, Plotly

More detailed information can be found in requirements.txt file.
<a name="file_structure"></a>
### File Structure
This is the file structure of the project:

├───app
│   └───templates
│   │   └───go.html
│   │   └───master.html
│   └───run.py
├───data
│   └───disaster_categories.csv
│   └───disaster_messages.csv
│   └───DisasterResponse.db
│   └───process_data.py
├───models
│   └───classifier.pkl
│   └───train_classifier.py
└───preparation
│   └───categories.csv
│   └───classifier.pkl
│   └───ETL Pipeline Preparation.ipynb
│   └───messages.csv
│   └───ML Pipeline Preparation.ipynb
└───screenshots
│   └───main_page.PNG
│   └───sample_input.PNG
│   └───sample_output.PNG
└───.flake8
└───README.md
└───requirements.txt

The most important files are:
* app/run.py: Launch the Flask app used to classify text messages.
* data/process_data.py: Responsible of ETL pipeline, extract, clean, transform and store the data in the SQLite database.
* model/train_classifier.py: Responsible of Machine Learning pipeline, load the data for training the model and save the model as .pkl file which can be consumed later from the predictions via web app.

<a name="installation"></a>
### Installing
* Clone the repository.

`git clone https://github.com/adriapa5/disaster-response-pipeline.git`

* Be sure that you have installed the required version of python or event better you have a proper conda enviroment ready with python 3.7+.
* Install the necessary libraries provided in requirements.txt file.
* Follow the instructions provided in the next section.

<a name="instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="material"></a>
### Additional Material

There are a couple of jupyter notebooks in **preparation** folder that can be useful to understand how the model is built step by step:
1. **ETL Preparation Notebook**: related to the implementation of the ETL pipeline
2. **ML Pipeline Preparation Notebook**: related to the implementation of the Machine Learning Pipeline

The **ML Pipeline Preparation Notebook** can be used to re-train or tune the model.

<a name="production"></a>
### It's Alive

This application has been deployed and you can play with it [here](https://disaster-response-pipeline-zkxgj.ondigitalocean.app/)

1. The main page shows some graphs about training dataset, provided by Figure Eight

![Main Page](screenshots/main_page.PNG)

2. This is an example of a message that can be introduced to test the performance of the model

![Sample Input](screenshots/sample_input.PNG)

3. After clicking **Classify Message**, the message will be categorized and the belonging message categories will be highlighted in green

![Sample Prediction](screenshots/sample_output.PNG)

<a name="acknowledgements"></a>
## Acknowledgements
* [Udacity](https://www.udacity.com/) for proposing this project as part of the Data Science Nanodegree Program.
* [Figure Eight](https://www.figure-eight.com/) for providing the data to train the model.
<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
