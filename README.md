# Disaster_Response

## Table of Contents

1. [Installations](#installation)
2. [Project Motivation](#motivation)
3. [Files Description](#files_description)
4. [How to Interact with the project](#interaction)
5. [Licensing, Authors, Acknowledgements, etc.](#licensing)

## Installation <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Project Motivation <a name="motivation"></a>
This project is the analysis of disaster data from Figure Eight to build a model for an API that classifies disaster messages in a web app.

## Files Description <a name="files_description"></a>
There are three components in this project:

1. ETL Pipeline (process_data.py)
Data cleaning pipeline in Python that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
  
2. ML Pipeline (train_classifier.py)
Machine learning pipeline in Python that:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
   
3. Flask Web App
Web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## How to Interact with the project <a name="interaction"></a>
In the web app, you can enter any message and then click on classify message to see how the model classifies it.

## Licensing, Authors, Acknowledgements, etc. <a name="licensing"></a>
The project was provided by Udacity and the data by Figure Eight.
