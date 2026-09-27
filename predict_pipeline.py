import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("Data Loading")
    train_data = pd.read_csv("data/raw_data/train.csv")
    test_data = pd.read_csv("data/raw_data/test.csv")

    X_train = train_data.drop(columns=["Will_Buy_EV", "id"], errors='ignore')
    y_train = train_data["Will_Buy_EV"]

    test_ids = test_data["id"]
    X_test = test_data.drop(columns=["id"], errors='ignore')

    number_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    category_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", number_transformer, make_column_selector(dtype_include='number')),
            ("categorical", category_transformer, make_column_selector(dtype_include='object'))
        ],
        remainder='drop'
    )
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42))
        ]
    )
    
    print("Model Training")
    model.fit(X_train, y_train)
    
    print("Making Predictions")

    probs = model.predict_proba(X_test)

    if 'Yes' in model.classes_:
        positive_class_idx = list(model.classes_).index('Yes')
    else:
        positive_class_idx = 1

    predictions = probs[:, positive_class_idx]

    print("Saving submission...")
    submission = pd.DataFrame({
        "id": test_ids,
        "Will_Buy_EV": predictions
    })

    print("Saving submission...")
    if os.path.exists('output/'):
        submission.to_csv("output/submission.csv", index=False)
        print("Pipeline completed successfully. Output saved to 'submission.csv'")
    else:
        os.makedirs('output/')
        submission.to_csv("output/submission.csv", index=False)
        print("Pipeline completed successfully. Output dir created and saved to 'submission.csv'")

if __name__ == "__main__":
    main()
