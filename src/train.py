"""Train the Ride Demand Forecaster baseline on its bundled synthetic dataset."""
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = json.loads((ROOT / 'project.json').read_text(encoding='utf-8'))
FEATURES = PROJECT['features']
TARGET = PROJECT['target']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data' / 'dataset.csv'))
    parser.add_argument('--limit', type=int, default=20000, help='Rows to use for a quick local run')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.data, nrows=args.limit)
    X = frame[FEATURES]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=None)
    model = KNeighborsRegressor(n_neighbors=7)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Project: {PROJECT['name']}")
    print(f'train_rows={len(X_train):,}  test_rows={len(X_test):,}')
    print(f'model={model.__class__.__name__}')
    # Task-appropriate holdout metrics
    print(f"mae={mean_absolute_error(y_test, pred):.4f}")
    print(f"rmse={mean_squared_error(y_test, pred) ** 0.5:.4f}")
    print(f"r2={r2_score(y_test, pred):.4f}")
    out = ROOT / 'models' / 'model.joblib'
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'target': TARGET, 'task_type': PROJECT['task_type'], 'project': PROJECT['name']}, out)
    print(f'Saved model: {out}')

if __name__ == '__main__':
    main()
