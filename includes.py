import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import joblib

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Project" / "data"
PROCESSED_DIR = DATA_DIR / "cleaned"
TRAIN_DIR = DATA_DIR / "train"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = PROJECT_DIR / "models"
XGB_REPORT_DIR = REPORTS_DIR / "xgboost"
XGB_MODEL_DIR = MODELS_DIR / "xgboost"
def load_data():

    print("Loading shared train-test split...")
    X_train = pd.read_csv(TRAIN_DIR / "X_train.csv")
    X_test = pd.read_csv(TRAIN_DIR / "X_test.csv")
    y_train = pd.read_csv(TRAIN_DIR / "y_train.csv")
    y_test = pd.read_csv(TRAIN_DIR / "y_test.csv")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test
