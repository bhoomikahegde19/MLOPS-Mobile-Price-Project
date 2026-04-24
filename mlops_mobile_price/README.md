# MLOps Mobile Price

`mlops_mobile_price` is a production-style machine learning project for mobile phone price prediction using the Kaggle mobile price dataset. The dataset target is `price_range`, so the regression models predict the ordinal price score while the classifier maps that score into `Low`, `Medium`, and `High`.

## Project Structure

- `pipeline/data`: dataset loading and preprocessing
- `pipeline/models`: linear regression, logistic regression, random forest, LSTM placeholder, and dummy NER
- `pipeline/training`: split creation, evaluation, AutoML selection, and training entrypoint
- `pipeline/registry`: versioned model saving
- `api`: FastAPI app, routes, dependencies, and services
- `experiments/mlruns`: local MLflow tracking store
- `docker/Dockerfile`: container definition
- `.github/workflows/ci.yml`: CI pipeline

## Setup

```powershell
cd mlops_mobile_price
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

```powershell
cd mlops_mobile_price
python -m pipeline.training.train
```

The loader will first look for `data/train.csv`, then try the Kaggle dataset, and finally use a public mirror of the same mobile-price training CSV if network access is available.

## Run The API

```powershell
cd mlops_mobile_price
uvicorn api.main:app --reload
```

## API Endpoints

- `GET /health`
- `POST /predict`
- `POST /classify`
- `POST /ner`

### Example Predict Request

```json
{
  "features": {
    "battery_power": 1200,
    "blue": 1,
    "clock_speed": 2.2,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 32,
    "m_dep": 0.5,
    "mobile_wt": 140,
    "n_cores": 4,
    "pc": 12,
    "px_height": 1200,
    "px_width": 1900,
    "ram": 3072,
    "sc_h": 14,
    "sc_w": 7,
    "talk_time": 15,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
  }
}
```

## MLflow

Run MLflow locally from the project root:

```powershell
cd mlops_mobile_price
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

## Docker

```powershell
cd mlops_mobile_price
docker build -f docker/Dockerfile -t mlops-mobile-price .
docker run -p 8000:8000 mlops-mobile-price
```
