# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    " load data from database
    "
    " Args:
    "   database_filepath: path of the file containing the databaase
    "
    " Returns:
    "   X: the messages
    "   Y: the categories
    "   Y.columns: the list of categories
    "
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Response', con=engine)
    X = df['message']
    Y = df.drop(['message','id','genre','original'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """
    " load data from database
    "
    " Args:
    "   text: the text to be tokenized
    "
    " Returns:
    "   tokens: the tokens extracted from the text
    "
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip()

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def build_model():
    """
    " build the ML pipeline to be used to predict the categories of the messages
    "
    " Args:
    "   None
    "
    " Returns:
    "   the ML pipeline
    "
    """
    # parameters found by using GridSearchCV
    svc = LinearSVC(random_state=0, C=0.5)
    multi_class_svc = OneVsRestClassifier(svc)
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.4)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(multi_class_svc))
                        ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    " evaluate the model by printing f1 score, precision and recall
    "
    " Args:
    "   model: the model to be tested
    "   X_test: The data to test against
    "   Y_test: the expected result of the prediction
    "   category_names: the names of the categories
    "
    " Returns:
    "   nothing
    "
    """
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred.columns = Y_test.columns
    for column in category_names:
        print(column + ':')
        print(classification_report(Y_test[column], y_pred[column], target_names=category_names))


def save_model(model, model_filepath):
    """
    " save the model in a pickle file
    "
    " Args:
    "   model: the model to be saved
    "   model_filepath: the path of the target file
    "
    " Returns:
    "   nothing
    "
    """
    try:
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
