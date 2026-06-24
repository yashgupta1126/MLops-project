# 🌍 MLOps End-To-End Machine Learning Pipeline-CICD

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Frontend: Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Containerized: Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI/CD: GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Deployment: Docker Swarm](https://img.shields.io/badge/Deployment-Docker%20Swarm-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)


## 🌐 Live App
The application is deployed using Docker Swarm with a fully automated GitHub Actions CI/CD pipeline.

👉 **Live Application:** http://80.225.233.115:8501/

An end-to-end **MLOps Image Classification Pipeline** that classifies natural scene images into six categories:

🏢 Buildings • 🌲 Forest • ❄️ Glacier • ⛰️ Mountain • 🌊 Sea • 🛣️ Street

The project covers the complete machine learning lifecycle including **data preprocessing, CNN model training, model serialization, Docker containerization, GitHub Actions CI/CD automation, and Docker Swarm deployment**.

---

## 🚀 Features

- 🧠 **Deep Learning Image Classification**
  - Custom CNN built using TensorFlow/Keras.
  - Trained on over **14,000+ images** across six scene categories.

- 📊 **Image Data Augmentation**
  - Rotation
  - Width/Height Shifts
  - Horizontal Flips
  - Vertical Flips

- 🌍 **Modern Streamlit Web Application**
  - Upload local images.
  - Predict images from URLs.
  - Interactive UI with real-time classification.

- 📈 **Confidence-Based Predictions**
  - Displays predicted class along with confidence score.

- 🐳 **Fully Containerized**
  - Dockerized application for reproducible deployments.

- ⚙️ **Automated CI/CD Pipeline**
  - GitHub Actions workflow.
  - Automatic Docker image build & push.
  - Automatic deployment on code push.

- ☁️ **Production Deployment**
  - Docker Swarm orchestration.
  - Rolling updates.
  - Resource limits and restart policies.

---

# 📂 Dataset Information

The model is trained on the Intel Image Classification Dataset containing six categories:

| Class | Description |
|---------|-------------|
| 🏢 Buildings | Urban buildings and architecture |
| 🌲 Forest | Dense forest landscapes |
| ❄️ Glacier | Snow and ice-covered glaciers |
| ⛰️ Mountain | Mountainous terrain |
| 🌊 Sea | Oceans, beaches, and water bodies |
| 🛣️ Street | Roads and city streets |

### Dataset Split

| Dataset | Images |
|----------|----------|
| Training | 14,034 |
| Testing | 3,000 |
| Prediction Set | 7,301 |

---

## 🧱 Tech Stack

| Layer | Technologies |
|---------|-------------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow 2.15, Keras |
| **Frontend** | Streamlit |
| **Data Processing** | NumPy, Pillow |
| **Model Training** | TensorFlow CNN |
| **Containerization** | Docker |
| **Orchestration** | Docker Swarm |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git & GitHub |

---

# 🧠 Model Architecture

The CNN consists of:

```text
Input Layer (150×150×3)

│
├── Conv2D (16 Filters, ReLU)
├── MaxPooling2D

├── Conv2D (32 Filters, ReLU)
├── MaxPooling2D

├── Conv2D (64 Filters, ReLU)
├── MaxPooling2D

├── Flatten

├── Dense (128 Units, ReLU)

└── Dense (6 Units, Softmax)
```

### Model Statistics

| Metric | Value |
|----------|----------|
| Total Parameters | 2,678,694 |
| Trainable Parameters | 2,678,694 |
| Input Size | 150 × 150 × 3 |
| Classes | 6 |

---

# 📈 Training Results

The model was trained for **5 epochs**.

| Epoch | Training Accuracy | Validation Accuracy |
|---------|-------------------|----------------------|
| 1 | 54.22% | 57.70% |
| 2 | 65.77% | 67.93% |
| 3 | 71.59% | 76.63% |
| 4 | 73.82% | 77.47% |
| 5 | 76.56% | 79.90% |

### Final Test Performance

```text
Test Accuracy : 79.90%
Test Loss     : 0.5515
```

---

# 🛠️ Project Structure

```text
PROJECT
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── .streamlit/
│   └── config.toml
│
├── newmodel1/
│   ├── assets/
│   ├── variables/
│   ├── fingerprint.pb
│   ├── keras_metadata.pb
│   └── saved_model.pb
│
├── seg_train/
├── seg_test/
│
├── Dockerfile
├── docker-stack.yml
├── myapp.py
├── newmodel.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ CI/CD Pipeline

The project follows a complete MLOps workflow:

### Step 1 — Code Push

```text
Developer
    ↓
GitHub Repository
```

### Step 2 — GitHub Actions

```text
GitHub Actions Trigger
    ↓
Build Docker Image
    ↓
Push Image to Docker Hub
```

### Step 3 — Deployment

```text
Docker Hub
    ↓
Docker Swarm Cluster
    ↓
Rolling Update
    ↓
Production Deployment
```

### Pipeline Features

- Automated builds
- Automated deployment
- Versioned Docker images
- Docker Hub integration
- SSH-based production deployment
- Rolling updates with Docker Swarm

---

# 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t location-classifier .
```

### Run Container

```bash
docker run -p 8501:8501 location-classifier
```

### Access Application

```text
http://localhost:8501
```

---

# 🚀 Docker Swarm Deployment

Initialize Docker Swarm:

```bash
docker swarm init
```

Deploy Stack:

```bash
docker stack deploy -c docker-stack.yml my-ml-app
```

Check Services:

```bash
docker service ls
```

Check Running Containers:

```bash
docker ps
```

---

# 🔧 Local Installation

### Clone Repository

```bash
git clone https://github.com/yashgupta1126/MLops-project.git

cd MLops-project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run myapp.py
```

---

# 📚 Usage Guide

### Upload an Image

1. Open the web application.
2. Upload a JPG, JPEG, or PNG image.
3. Click **Classify Image**.
4. View predicted category and confidence score.

### Predict Using Image URL

1. Paste a valid image URL.
2. Click **Classify Image**.
3. View model prediction.

### Example Categories

- Forest
- Glacier
- Mountain
- Buildings
- Sea
- Street

---

## 📄 License
This project is licensed under the MIT License.

---

## 🤝 Contributions
Feel free to fork, raise issues, or submit PRs to improve this project!

---

## 📝 Author
**Yash Gupta** | IIT Kharagpur Mechanical Engineering

Email: [yg291557@gmail.com]