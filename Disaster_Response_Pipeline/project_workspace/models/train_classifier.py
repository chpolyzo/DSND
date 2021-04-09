import sys
# import necessary libraries to load data from the database
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# import necessary libraries
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download(['punkt', 'words', 'stopwords', 'averaged_perceptron_tagger', 'wordnet'])

#import necessary libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer

import time
import pickle

def load_data(database_filepath):
    '''
    Input: 
    Database filepath
    Output: 
    multi-labels Y which is a dataframe holding information about our 34 categories to classify,
    a numpy array X keeping text to be classified
    
    Description:
    Provides curated data from sql database to X numpy.arrary and y dataframe multiclass variables
    '''
    engine = 'sqlite:///' + database_filepath
    df = pd.read_sql_table('response_table', con=engine)
    X = df.loc[:,'message'].values
    Y = df.iloc[:,4:]
    
    return X, Y


def tokenize(text):
    """
    Input: 
    a text string found in each reccord (str)
    Output:
    a list of stems 
    
    Desscription:
    Function that cterates stems - word tokens
    1. replaces urls with the 'url' string
    2. replaces punctuation marks with white spaces
    3. creates lists of words out of the initial text
    4. assigns Parts of speech to every word
    5. reduces words to their root form by specifying verb parts of speech
    6. reduces words to their stems - not necessary words to be understood by humans
    
    
    """
    # regex pattern to identify an url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replace urls with a 'url' string
    text = re.sub(url_regex, 'url', text)
    # text normalization - remove punctuation and lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text to words
    words = [w for w in word_tokenize(text) if w not in stopwords.words("english")]
    # assign "Parts of Speech": POS to every word - words output is a tupple
    words = pos_tag(words)
    # Reduce words to their root form by specifying Part of Speech: verb
    lemmed = [WordNetLemmatizer().lemmatize(w[0], pos = 'v') for w in words]
    # Reduce words to their stems - that is their root form not exactly a word to be understood 
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    
    return stemmed

def split_n_train(X, Y):
    """
    input: 
    multi-labels Y which is a dataframe holding information about our 34 categories to classify,
    a numpy array X keeping text to be classified
    a tokenizer function to tokenize our text
    
    output:
    a trained classification model
    X_train: 60% of the X  array for trainning purposes
    X_test: remaining 40% of the X array for testing purposes
    y_train: 60% rows of the Y dataframe to train our classifier
    y_test: 40% remaining 
    ])
    
    Description 
    splits and trains the classifier
    """
    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    start = time.time()/60
    # train classifier
    fitted = pipeline.fit(X_train, y_train)
    stop = time.time()/60
    print(f"Model calculation time: {round(stop - start)} minutes") 
    
    return fitted, X_train, X_test, y_train, y_test

def build_model():
    """
    Input:
    no input
    
    output:
    improoved model
    
    Description:
    a cross validated fitted model with improved parameters using GridSearch.
    In our case the number of trees in the forest.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    parameters = {'clf__n_estimators': [50]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def eval_model(X_test, fitted_model):
    """
    input:
    X_test = 60% of the X array to test our model
    The fitted model we have trained on our classifier
    
    output:
    y_pred = predicted outputs that indicate what
    kind of request each text is refering to our
    of our 34 categories we train
    
    Descriprion:
    takes a trained model and aplies the test dataset
    """
    
    start = time.time()/60
    y_pred = fitted_model.predict(X_test)
    stop = time.time()/60
    print(f"Model evaluation time: {round(stop - start)} minutes") 
    
    return y_pred

def evaluate_model(model, X_test, Y_test, category_names):
    pass

def display_results(y_test, y_pred, fitted_model, *cvd):
    '''
    input:
    y_pred = predicted outputs that indicate what
    kind of request each text is refering to our
    of our 34 categories we train
    
    output:
    displays the accuracy of each predictor of our classifier
    displays Pipeline Parameters
    displays the best parameters for a model run with Grid Search
    
    '''
    accuracy = (y_pred == y_test).mean()
    
    print("Accuracy for each predictor:")
    print(accuracy)
    
    # get the params of the fitted model
    print("Get Pipeline parameters")
    for key in fitted_model.get_params().keys():
        print(key)
        
    # check best parameters in cross validated models
    for cv in cvd:
        print("\nBest Parameters are:", cv.best_params_)
        
        
def save_model(fitted_model, model_filepath):
    '''
    Input:
    the fitted model
    model_filepath (str) is the path of the pickle file to be saved
    '''

    with open(model_filepath, 'wb') as f:
        pickle.dump(fitted_model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        print('Building and training model...')
        fitted_model, X_train, X_test, y_train, y_test = split_n_train(X, Y)
        
        print('Evaluating model...')
        y_pred = eval_model(X_test, fitted_model)
        display_results(y_test, y_pred, fitted_model)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(fitted_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()