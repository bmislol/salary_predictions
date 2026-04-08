import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model(data_path):
    # we load the processed data
    df = pd.read_csv(data_path)

    # split features (X) and target (y)
    X = df.drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    # train test split (will use 80% for training and 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # we will train 3 different models and compare their performance
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    result = {}

    # create directories needed for here
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print(f"{'Model':<20} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 45)

    for name, model in models.items():
        # train
        model.fit(X_train, y_train)

        # predict
        predictions = model.predict(X_test)

        # evaluate
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        result[name] = {"model": model, "mae": mae, "r2": r2, "preds": predictions}

        print(f"{name:<20} | {mae:<10.2f} | {r2:10.4f}")

    # visualizing the result (for desicion tree)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=result['Decision Tree']['preds'])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel("Actual Salary (USD)")
    plt.ylabel("Predicted Salary (USD)")
    plt.title("Decision Tree: Actual vs Predicted Salaries")
    plt.savefig('results/model_performance.png')
    print("\n📈 Performance graph saved to results/model_performance.png")

    # use joblib to save the model
    best_model = result["Decision Tree"]["model"]
    joblib.dump(best_model, 'models/salary_model.joblib')
    print("💾 Decision Tree model saved to models/salary_model.joblib")

if __name__ == "__main__":
    DATA_PATH = 'data/processed_salaries.csv'

    if os.path.exists(DATA_PATH):
        train_model(DATA_PATH)
    else:
        print(f"❌ Processed data not found at {DATA_PATH}. Please run the cleaning.py script first.")