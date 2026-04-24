# 📱 MLOps Mobile Price Prediction System

## 🚀 Overview

This project is a **production-grade MLOps pipeline** for predicting mobile phone prices using machine learning.
It demonstrates the **complete lifecycle of an ML system** — from data preprocessing to deployment.

---

## 🧠 Models Used

* 📊 **Linear Regression (Main Model)**
* 🔍 Logistic Regression (Classification)
* 🌲 Random Forest (Comparison Model)
* 🔁 LSTM (Basic Placeholder)
* 🏷️ NER (Named Entity Recognition - Dummy)

---

## 🏗️ Project Architecture

```bash
mlops_mobile_price/
│
├── pipeline/
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # ML models
│   ├── training/          # Training & evaluation
│   ├── registry/          # Model versioning
│
├── api/
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic
│
├── docker/                # Containerization
├── .github/workflows/     # CI/CD pipeline
```

---

## ⚙️ Key Features

✅ End-to-end ML pipeline
✅ Multi-model training & comparison
✅ Automated model selection (AutoML logic)
✅ MLflow experiment tracking
✅ FastAPI backend for serving models
✅ Docker-ready deployment
✅ CI/CD pipeline using GitHub Actions

---

## 📊 Dataset

* 📂 Source: Kaggle
* 🔗 Mobile Phone Price Prediction Dataset

Features include:

* RAM
* Storage
* Battery
* Camera
* Processor

---

## 🔄 Workflow

```text
Data → Preprocessing → Model Training → Evaluation → Best Model Selection → API Deployment
```

---

## 🌐 API Endpoints

| Endpoint    | Method | Description                              |
| ----------- | ------ | ---------------------------------------- |
| `/predict`  | POST   | Predict mobile price (Linear Regression) |
| `/classify` | POST   | Categorize phone price                   |
| `/ner`      | POST   | Extract entities (dummy)                 |
| `/health`   | GET    | Check API status                         |

---

## ▶️ Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bhoomikahegde19/MLOPS-Mobile-Price-Project.git
cd MLOPS-Mobile-Price-Project
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the API

```bash
uvicorn api.main:app --reload --port 8001
```

---

### 5️⃣ Open API Docs

👉 http://127.0.0.1:8001/docs

---

## 📥 Example Input

```json
{
  "features": [8, 128, 5000, 48, 1.8]
}
```

---

## 📤 Example Output

```json
{
  "predicted_price": 25000.0
}
```

---

## 🐳 Docker Support

```bash
docker build -t mlops-app .
docker run -p 8000:8000 mlops-app
```

---

## 🔁 CI/CD Pipeline

* Automated testing
* Model training on push
* Deployment-ready setup

---

## 📈 Future Improvements

* 🔥 Real LSTM implementation
* 🎨 Frontend dashboard (React/Streamlit)
* 📊 Model monitoring & drift detection
* ☁️ Cloud deployment (AWS / GCP)

---

## 👩‍💻 Author

**Bhoomika Hegde**
Artificial Intelligence & Machine Learning Student
Dayananda Sagar College of Engineering

---

## ⭐ Acknowledgements

* Kaggle Dataset
* Scikit-learn
* FastAPI
* MLflow

---

## 💡 Final Note

This project demonstrates how machine learning models can be **designed, tracked, deployed, and scaled** using modern MLOps practices — making it closer to real-world industry systems.

---
