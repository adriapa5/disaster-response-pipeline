import sys
import nltk
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def load_data(database_filepath):
    """
    Load Data from the Database Function

    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        y -> a dataframe containing labels
        category_names -> List of categories name
    """

    # load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    table_name = database_filepath.replace(".db", "")
    df = pd.read_sql_table(table_name, engine)

    # remove child alone as it has all zeros only
    df = df.drop(["child_alone"], axis=1)

    # the related field with value 2 is very small and this can cause errors.
    # This is why I decided to replace the value 2 of this column to 1,
    # considering a valid response and because it's the majority class
    df["related"] = df["related"].map(lambda x: 1 if x == 2 else x)

    # prepare the output
    X = df["message"]
    y = df.iloc[:, 4:]
    category_names = y.columns.values
    return X, y, category_names


def tokenize(text, url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function

    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        tokens -> List of tokens extracted from the provided text
    """

    # replace all urls with a urlplaceholder string
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # extract the word tokens from the provided text
    words = nltk.word_tokenize(text)

    # lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # list of clean tokens
    tokens = [lemmatizer.lemmatize(w).lower().strip() for w in words]
    return tokens


# build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True
        return False

    # given it is a tranformer we can return the self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build Model function

    Output:
        Trained Model after performing grid search and built using Scikit ML Pipeline
        that process text messages and apply a classifier.

    """
    # model pipeline
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "count_vectorizer",
                                        CountVectorizer(tokenizer=tokenize),
                                    ),
                                    ("tfidf_transformer", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("starting_verb_transformer", StartingVerbExtractor()),
                    ]
                ),
            ),
            ("classifier", MultiOutputClassifier(AdaBoostClassifier())),
        ]
    )

    # hyper-parameter grid
    parameters = {
        "classifier__estimator__learning_rate": [0.01, 0.02, 0.05],
        "classifier__estimator__n_estimators": [10, 20, 40],
    }

    # create model
    model = GridSearchCV(pipeline, param_grid=parameters, scoring="f1_micro", n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function

    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)

    Arguments:
        model -> trained model
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print("Accuracy: {}".format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
    Save Pipeline function

    This function saves trained model as Pickle file, to be loaded later.

    Arguments:
        model -> GridSearchCV or Scikit Pipeline model object
        model_filepath -> destination path to save .pkl file

    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
