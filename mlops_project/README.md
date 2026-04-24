# Mobile Phone Price Prediction MLOps Project

This repository contains a production-style MLOps system for end-to-end mobile phone price prediction. It includes data ingestion, preprocessing, multiple machine learning models, AutoML-style model comparison, MLflow experiment tracking, model registry/versioning, FastAPI inference endpoints, Docker packaging, and a GitHub Actions CI pipeline.

## Architecture Overview

- `pipeline/data`: dataset loading, target inference, preprocessing, and train/test splitting
- `pipeline/models`: five models exactly as requested
- `pipeline/training`: dataset helpers, evaluation, AutoML orchestration, and full training pipeline
- `pipeline/registry`: model promotion and versioning
- `api`: FastAPI app, routes, dependency wiring, and inference services
- `experiments/mlruns`: local MLflow tracking backend
- `docker/Dockerfile`: containerized API runtime
- `.github/workflows/ci.yml`: lint, train, and Docker build automation

## Models Implemented

1. Linear Regression: main regression model for mobile price prediction
2. Logistic Regression: price-band classification (`Low`, `Medium`, `High`)
3. Random Forest: regression benchmark for model comparison
4. LSTM: sequence-style tabular regressor built with PyTorch
5. NER Model: lightweight entity extractor for mobile text fields such as brand, RAM, and storage

## Dataset Handling

The pipeline is designed for the Kaggle dataset at:
`https://www.kaggle.com/datasets/dewangmoghe/mobile-phone-price-prediction`

Place the CSV file at one of these locations:

- `mlops_project/data/mobile_phone_price_prediction.csv`
- `mlops_project/data/mobile_phone_price.csv`
- `mlops_project/data/mobile_prices.csv`
- `mlops_project/data/train.csv`

You can also set a custom location:

```powershell
$env:DATASET_PATH="C:\path\to\your\dataset.csv"
```

If no CSV is present, the loader generates a realistic synthetic mobile-price dataset so the project still runs end to end.

## Setup

```powershell
cd mlops_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training

```powershell
cd mlops_project
python -m pipeline.training.train
```

Training will:

- load and preprocess the dataset
- train all five models
- compare regression candidates automatically
- log runs and artifacts to MLflow in `experiments/mlruns/`
- promote versioned artifacts into `artifacts/registry/`

## MLflow

Start the MLflow UI locally from the project root:

```powershell
cd mlops_project
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

## API

Start the API:

```powershell
cd mlops_project
uvicorn api.main:app --reload
```

### Health Check

```bash
curl -X POST http://127.0.0.1:8000/health
```

### Predict Price

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\":{\"brand\":\"Samsung\",\"storage_gb\":128,\"ram_gb\":8,\"battery_mah\":5000,\"screen_size_in\":6.5,\"camera_mp\":64,\"refresh_rate_hz\":120,\"processor_score\":320,\"has_5g\":1,\"dual_sim\":1}}"
```

### Classify Price Band

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d "{\"features\":{\"brand\":\"Samsung\",\"storage_gb\":128,\"ram_gb\":8,\"battery_mah\":5000,\"screen_size_in\":6.5,\"camera_mp\":64,\"refresh_rate_hz\":120,\"processor_score\":320,\"has_5g\":1,\"dual_sim\":1}}"
```

### Extract Entities

```bash
curl -X POST http://127.0.0.1:8000/ner \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Samsung Galaxy with 8GB RAM and 256GB storage\"}"
```

## Docker

Build and run:

```powershell
cd mlops_project
docker build -f docker/Dockerfile -t mobile-price-mlops .
docker run -p 8000:8000 mobile-price-mlops
```

## CI/CD

The GitHub Actions workflow performs:

- dependency installation
- linting with `ruff`
- end-to-end training execution
- Docker image build validation

## Notes

- `POST /predict` uses the Linear Regression model.
- `POST /classify` uses the Logistic Regression model.
- `POST /ner` uses the NER service layer.
- `GET /stream` exposes the latest AutoML summary as a streaming response.
