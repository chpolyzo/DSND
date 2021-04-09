# Disaster Response Pipeline Project

### Motivation 
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. 
The dataset provided by Figure Eight is a pre-labelled dataset containing tweets and messagess from disaster events such as Floods, Hurricanes and Earthquakes. 
The project aims to support Decision Making and hopefully Humanitarian Relief organizations such as the 
- [United Nations For Disaster Risk Reduction](https://www.undrr.org/),
- [European civil protection and humanitarian aid operations](https://ec.europa.eu/echo/what/civil-protection/emergency-response-coordination-centre-ercc_en),
- [Federal Emergency Management Agency](https://www.fema.gov/),
- [Other local civil protection authorities](http://www.protezionecivile.gov.it/)
to name a few

Machine Learning is crucial to act fast and take informed decisions in order to provide the right help where is needed at the time needed and save lived

A comprehensive planning against Disasters generally include 

- 1. Mitigation
- 2. Preparedness
- 3. Response 
- 4. Recovery

If we were to include a Natural Language Processing (NLP) model to categorize messages life would be much less dangerous. 
Unfortunatelly very little has been achieved in this direction.

This project is structured in the following key sections:

- Processing data, 
- Building an ETL pipeline (Extract Transform Load pipeline) to extract data from source, 
- Clean the data and save them in a SQLite DB
- Build a machine learning pipeline to train the which can classify text message in various categories
- Run a web app which can show model results in real time

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files Description

'''
- README.md: read me file
- class_notebooks
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
- project_workspace
	- \app
		- run.py: flask file to run the app
	- \templates
		- master.html: main page of the web application 
		- go.html: result web page
	- \data
		- disaster_categories.csv: categories dataset
		- disaster_messages.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL process
	- \models
		- train_classifier.py: classification code
- miscellanious_data
	- Other Related Information

'''


- The `app` folder keeps the code for the application user interface
#### Here you have some screenshots of the app to classify messages from a text
##### 1. Distributions of Genres
![DisasterResponce](Disaster_Responce.png)
##### 2. Here you have the classification User Interface (UI)
![Messages Classifications](Disaster_Responce_classify_message.png)
##### 3. Running the app from the terminal
![Running the app from the terminal](Disaster_Responce_Project_complete.png)

### Python libraries

Python libraries used for this project are:
- [Machine learning library scikit-learn](https://scikit-learn.org/)
- [Data elaboration library Pandas](https://pandas.pydata.org/)
- [Data elaboration library NumPy](https://numpy.org/)
- [NLTK - Natural Language Processing](https://www.nltk.org/)
- [SQLlite Database library SQLalchemy](https://www.sqlalchemy.org/)
- [Python object serialization Pickle](https://docs.python.org/3/library/pickle.html)
- [Web App library Flask](https://flask.palletsprojects.com/)
- [Python Graphing Library Plotly](https://plotly.com/python/)

### Acknowledgements
Thanks to all fellow udacitians that have worked on this project and shared their work so that we can get some hints and inspirations
- https://github.com/iris-theof/Disaster_response_pipeline
- https://github.com/canaveensetia/udacity-disaster-response-pipeline
- https://github.com/Mcamin/Disaster-Response-Pipeline
- https://github.com/sousablde/Disaster-Response-Pipeline
- https://github.com/Chintan5384/Disaster-Response-Pipeline


I want to say a special thank you to my wonderful mentors, reviewers and instructors at Udacity building up restlessly such amazing content every day, in every form they might have apeared!
It is amazing being able to communicate in such a special way. 



