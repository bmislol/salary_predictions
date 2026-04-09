import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def clean_data(file_path, output_path):
    # so we can see 50 columns at once, better for debugging
    pd.set_option('display.max_columns', 50) 

    # create the dataframe and print out the original stats we are working with
    df = pd.read_csv(file_path)
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Head of the original DataFrame:\n{df.head()}")

    # drop redundant columns (3ana salary in usd so we dont need be2e l salaries, w kamen mana eyzin index fa we drop this column)
    cols_to_drop = ['Unnamed: 0', 'salary', 'salary_currency']
    df = df.drop(columns=cols_to_drop)
    print(f"DataFrame shape after dropping index, salary and salary_currency columns:\n {df.shape}")
    print(f"DataFrame head after dropping index, salary and salary_currency columns:\n {df.head()}")

    # we can check for any missing values
    total_missing = df.isnull().sum()
    print(f"Total missing values in each column:\n{total_missing}")

    # we will use LabelEnconder to convert strings to unique integers (we will also save the enconders for later use in the API)
    encoders = {}
    categorical_columns = [
        'experience_level',
        'employment_type',
        'job_title',
        'employee_residence',
        'company_size',
        'company_location'
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, 'models/encoders.joblib')
    print("💾 Encoders saved to models/encoders.joblib")

    print(f"\nDataFrame head after encoding categorical variables:\n{df.head()}")

    # lets check if everything is okay before saving the new data
    print("\nF--- Final Data Types ---")
    print(df.info())

    # check the distribution of our target variable
    print("\n--- Salary Statistics ---")
    print(df['salary_in_usd'].describe())

    # lets save it to a new csv file this way we always have the original for debugging
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    # paths we need for the function
    INPUT = 'data/ds_salaries.csv'
    OUTPUT = 'data/processed_salaries.csv'

    # make sure that they exist (kermel exception handling)
    if not os.path.exists('data'):
        os.makedirs('data')
        print("📁 Directory 'data' created.")

    if not os.path.exists(INPUT):
        print(f"❌ Error: {INPUT} not found. Please add the dataset and run again.")
    else:
        clean_data(INPUT, OUTPUT)
    
