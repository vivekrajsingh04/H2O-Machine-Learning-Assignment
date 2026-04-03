import h2o
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
import os

def main():
    # 1. Initialize H2O cluster
    print("Initializing H2O cluster...")
    h2o.init()

    # --- Data Understanding and Preprocessing ---
    # We will use the Iris dataset from UCI (downloading via pandas and saving to CSV to mimic local loading)
    print("Downloading and preparing dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(url, names=columns)
    df.to_csv("iris.csv", index=False)
    
    # - Load dataset in H2O
    print("Loading dataset into H2O...")
    data = h2o.import_file("iris.csv")
    
    # - Handle missing values (Iris has none, but we demonstrate imputation)
    # Just in case, we impute numeric columns with median
    for col in columns[:-1]:
        data.impute(col, method="median")
        
    # - Data transformation: ensure the target is a categorical factor (for Classification)
    data["class"] = data["class"].asfactor()

    print("Data Preview:")
    print(data.head(3))
    
    # --- Introduction to Machine Learning ---
    # - Identify problem type: Multi-class Classification
    # - Split dataset
    print("Splitting dataset into train and test...")
    train, test = data.split_frame(ratios=[0.8], seed=1234)
    x = data.columns[:-1]
    y = data.columns[-1]
    
    # - Build basic model (Random Forest)
    print("Building Basic Model (Random Forest)...")
    rf_model = H2ORandomForestEstimator(ntrees=50, max_depth=5, seed=1234)
    rf_model.train(x=x, y=y, training_frame=train)
    
    # --- Mathematical Optimization ---
    # - Identify loss function: Multinomial loss (logloss) for multi-class classification
    # - Evaluate performance
    rf_perf = rf_model.model_performance(test)
    print("Random Forest Test Performance (Logloss):", rf_perf.logloss())
    print("Random Forest Test Performance (Error):", rf_perf.mean_per_class_error())
    
    # --- Neural Networks ---
    # - Build simple neural network
    print("Building Simple Neural Network...")
    nn_simple = H2ODeepLearningEstimator(hidden=[10], epochs=20, activation="Tanh", seed=1234)
    nn_simple.train(x=x, y=y, training_frame=train)
    
    # - Evaluate model
    nn_simple_perf = nn_simple.model_performance(test)
    print("Simple NN Test Performance (Logloss):", nn_simple_perf.logloss())
    
    # --- Deep Learning ---
    # - Use H2O Deep Learning & Tune parameters via Grid Search
    print("Tuning Deep Learning Model via Grid Search...")
    dl_params = {
        'hidden': [[20, 20], [50, 50]],
        'epochs': [50, 100],
        'l1': [1e-5, 1e-3]
    }
    
    dl_grid = H2OGridSearch(model=H2ODeepLearningEstimator(activation="RectifierWithDropout", hidden_dropout_ratios=[0.2, 0.2], seed=1234),
                            hyper_params=dl_params,
                            grid_id="dl_grid",
                            search_criteria={'strategy': 'Cartesian'})
    
    dl_grid.train(x=x, y=y, training_frame=train)
    
    # Get the best model
    best_dl_model = dl_grid.get_grid(sort_by='logloss', decreasing=False).models[0]
    best_dl_perf = best_dl_model.model_performance(test)
    print("Best Tuned DL Model Config:", best_dl_model.params.get('hidden'), "epochs:", best_dl_model.params.get('epochs'))
    print("Best Tuned DL Test Performance (Logloss):", best_dl_perf.logloss())
    
    # --- Results and Analysis ---
    # Compare models
    print("\n========= MODEL PERFORMANCE SUMMARY =========")
    print(f"Basic Model (Random Forest) Logloss:  {rf_perf.logloss():.4f}")
    print(f"Simple Neural Network Logloss:        {nn_simple_perf.logloss():.4f}")
    print(f"Tuned Deep Learning Model Logloss:    {best_dl_perf.logloss():.4f}")
    
    # Identify best model
    loglosses = {
        "Random Forest": rf_perf.logloss(),
        "Simple NN": nn_simple_perf.logloss(),
        "Tuned DL": best_dl_perf.logloss()
    }
    best_model_name = min(loglosses, key=loglosses.get)
    print(f"\nBest Performing Model: {best_model_name} with logloss of {loglosses[best_model_name]:.4f}")

if __name__ == "__main__":
    main()
