# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Here you have some screenshots of the app to classify messages from a text
##### Distributions of Genres
![DisasterResponce](Disaster_Responce.png)
##### Here you have the classification User Interface (UI)
![Messages Classifications](Disaster_Responce_classify_message.png)
![Running the app from the terminal](Disaster_Responce_Project_complete.png)
