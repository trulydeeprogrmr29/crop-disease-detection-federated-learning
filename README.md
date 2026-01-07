# Crop Disease Detection using Federated Learning

## About the Project
Crop diseases are a major problem in agriculture and can reduce crop yield if not detected early.  
In this project, I am building a **crop disease detection system** using **Federated Learning**.

The main idea is to train a machine learning model using data from multiple sources (clients) **without sharing the actual image data**.  
Each client trains the model locally and only shares model updates with the server.

This project is created as a **learning project** to understand:
- Federated Learning
- Image classification
- Privacy-aware machine learning

---
## 🎯 Key Objectives
- Detect crop diseases from leaf images
- Preserve farmer data privacy using Federated Learning
- Enable decentralized model training across multiple clients
- Build a scalable and real-world deployable ML pipeline

## 🧠 Core Concepts Used
- **Federated Learning (FL)** – Distributed model training without centralized data
- **Convolutional Neural Networks (CNNs)** – Image feature extraction and classification
- **Computer Vision** – Leaf image analysis
- **Privacy-Preserving Machine Learning**
- **Client–Server Architecture**

## Why Federated Learning?
In traditional machine learning, all data is collected in one place.  
But in real life, especially in agriculture, sharing data is not always possible.
✔ Data privacy for farmers
✔ No centralized data storage
✔ Scalable across regions
✔ Real-world compliant (GDPR-like constraints)

Federated Learning helps by:
- Keeping data on the client side
- Improving privacy
- Allowing multiple users to train a shared model

---

## What This Project Does
- Uses leaf images to detect crop diseases
- Trains a CNN model on each client
- Uses a federated server to combine model updates
- Evaluates the global model after training

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- PyTorch / TensorFlow (choose one)
- NumPy
- OpenCV

### Federated Learning
- Flower (FL Framework) / Custom FL Implementation

### Environment & Tooling
- **uv** – Ultra-fast Python package & environment manager
- Git & GitHub

## Project Structure

crop-disease-fl/
├── client/           # Client-side code

│   ├── main.py      # Client entry point

│   ├── trainer.py   # Local training logic
			
│   └── data_loader.py

├── server/          # Server-side code

│   ├── main.py      # Server entry point

│   ├── aggregator.py # Model aggregation

│   └── utils.py

├── shared/          # Shared components

│   ├── models.py    # Neural network architectures

│   └── config.py    # Shared configuration

├── datasets/        # Dataset utilities

├── configs/         # Configuration files

├── tests/           # Test suite

└── scripts/         # Utility scripts


🏗️ Project Architecture
```mermaid
flowchart TD
    Server[Federated Server] -->|Global Model| Client1
    Server -->|Global Model| Client2
    Server -->|Global Model| Client3
    
    subgraph Client1[Client 1 - Farm A]
        direction TB
        CM1[Local Model] --> TD1[Private Crop Images<br/>Disease: Leaf Rust, Blight, etc.]
        TD1 --> Train1[Local Training]
        Train1 --> Update1[Model Updates]
    end
    
    subgraph Client2[Client 2 - Farm B]
        direction TB
        CM2[Local Model] --> TD2[Private Crop Images<br/>Disease: Blight, Spot, etc.]
        TD2 --> Train2[Local Training]
        Train2 --> Update2[Model Updates]
    end
    
    subgraph Client3[Client N - Farm N]
        direction TB
        CM3[Local Model] --> TD3[Private Crop Images<br/>Disease: Various]
        TD3 --> Train3[Local Training]
        Train3 --> Update3[Model Updates]
    end
    
    Update1 -->|Encrypted Updates| Agg[Aggregator<br/>FedAvg / FedProx / FedAdam]
    Update2 -->|Encrypted Updates| Agg
    Update3 -->|Encrypted Updates| Agg
    
    Agg -->|New Global Model| Server
    
    style Server fill:#e3f2fd
    style Client1 fill:#f3e5f5
    style Client2 fill:#f3e5f5
    style Client3 fill:#f3e5f5
    style Agg fill:#e8f5e8

## Dataset
This project uses the **PlantVillage dataset**.  
The dataset contains images of healthy and diseased crop leaves.

To simulate federated learning, the dataset is divided into multiple parts and assigned to different clients.

---

## Setup Instructions (Using uv)

### 1. Clone the repository
```bash
git clone https://github.com/trulydeeprogrmr29/crop-disease-detection-federated-learning.git
cd crop-disease-detection-federated-learning


##Create virtual environment

uv venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows


##Install dependencies

uv pip install -r requirements.txt

How to Run the Project
Start the server
python server.py

Start clients (open separate terminals)
python client.py

Current Status
This project is still under development and improvements will be added step by step.



Future Improvements

Improve model accuracy
Basic CNN model implemented
Federated learning setup using Flower
Simple evaluation metrics
Add more clients
Test with different crops
Add better evaluation and visualization



Learning Outcome

Through this project, I am learning:
How federated learning works
How CNNs are used for image classification
How distributed training is done
How to structure a machine learning project


📊 Model Evaluation

Accuracy
Precision, Recall, F1-score
Comparison between:
Centralized Learning
Federated Learning

🤝 Contributing

Contributions are welcome!
Feel free to open issues, suggest improvements, or submit pull requests.
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request


📜 License

This project is licensed under the MIT License.
