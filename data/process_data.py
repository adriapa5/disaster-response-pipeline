import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data with Categories Function

    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Clean Data Function

    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [category_name.split("-")[0] for category_name in row.values[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # remove child alone as it has all zeros only
    categories = categories.drop(["child_alone"], axis=1)

    # the related field with value 2 is very small and this can cause errors.
    # This is why I decided to replace the value 2 of this column to 1,
    # considering a valid response and because it's the majority class
    categories["related"] = categories["related"].map(lambda x: 1 if x == 2 else x)

    # drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    cleaned_df = pd.concat([df, categories], join="inner", axis=1)
    return cleaned_df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function

    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """

    engine = create_engine("sqlite:///" + database_filename)
    table_name = database_filename.replace(".db", "")
    df.to_sql(table_name, engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
