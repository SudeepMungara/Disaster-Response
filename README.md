# Disaster-Response

1.Project Overview

In this project I've built a disaster response pipeline using data engineering skill on Appen dataset and a classification model using Flask API to classify disaster messages.

2. File Structure 
      - app<br>
      | - template <br>
      | |- master.html  # main page of web app <br>
      | |- go.html  # classification result page of web app <br>
      |- run.py  # Flask file that runs app<br>

      - data <br>
      |- disaster_categories.csv  # data to process <br>
      |- disaster_messages.csv  # data to process<br>
      |- process_data.py #ETL script to transform the data <br> 
      |- DisasterResponse.db   # database to save clean data to <br>

      - models <br>
      |- train_classifier.py #Training script to classify disaster messages <br>
      |- classifier.pkl  # saved model <br>


3.Instructions on running this project <br>

    a. Run the following commands in the project's root directory to set up your database and model.

        - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        - To run ML pipeline that trains classifier and saves
            `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    b. Run the command in the `app` directory: `python run.py`

    c. You can visit the webpage here http://0.0.0.0:3001/
