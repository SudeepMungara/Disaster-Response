import sys
import pandas as pd
import re
import nltk
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
        Load data from database
    Args:
        database_filepath: database file
    Returns:
        X: Features dataframe
        Y: Target dataframe
        category_names: labels 
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X,Y,category_names

def tokenize(text):
    '''
    Tokenize given text
    Args:
        text: message text
    Return:
        clean_tokens: clean tokens fromm text
    '''
    text =  ''.join([c for c in text if c not in punctuation])
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    words = [tok for tok in tokens if tok not in stopwords.words("english")]
    
    clean_tokens = []
    for word in words:
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
        Classification model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((AdaBoostClassifier())))
    ])

    parameters = {'tfidf__norm':['l2','l1'],'clf__estimator__learning_rate' :[0.1, 0.5, 1, 2]}

    cv = GridSearchCV(pipeline, parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model evaluation
        Args:
            model: Trained model
            X_test: Test features
            Y_test: Test labels
            category_names: labels 
    '''
    y_pred_cv = model.predict(X_test)
    for index, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred_cv[:, index]))


def save_model(model, model_filepath):
    '''
        Save model to specified path
        Args:
            model: model
            model_filepath: File path to dump pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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