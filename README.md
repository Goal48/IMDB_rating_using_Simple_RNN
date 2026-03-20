# 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A deep learning project that performs **binary sentiment classification** (Positive / Negative) on IMDB movie reviews using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras. The trained model is deployed as an interactive web app using **Streamlit**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🧠 Overview

This project uses the **IMDB dataset** (50,000 movie reviews) to train a Simple RNN model that classifies reviews as either **Positive** or **Negative**. It covers the full ML pipeline — from data preprocessing and model training to saving the model and serving it via a Streamlit web interface.

---

## 🚀 Demo

> Enter any movie review in the Streamlit app and instantly get a **Positive** or **Negative** prediction along with a confidence score.

---

## 📁 Project Structure
```
IMDB_rating_using_Simple_RNN/
│
├── IMDB.ipynb            # Model training notebook
├── Embedding.ipynb       # Word embedding exploration
├── Prediction.ipynb      # Prediction testing notebook
├── main.py               # Streamlit web application
├── imdb_rnn_model.h5     # Saved trained model
└── README.md
```

---

## ⚙️ How It Works

1. **Data Loading** — The IMDB dataset is loaded directly from `tensorflow.keras.datasets`, pre-tokenized with a vocabulary of the top 10,000 words.
2. **Preprocessing** — Reviews are padded/truncated to a fixed length of 500 tokens using `sequence.pad_sequences`.
3. **Model Training** — A Simple RNN model with an Embedding layer is trained for binary classification.
4. **Prediction** — User-inputted text is tokenized, encoded, padded, and passed through the model to produce a sentiment label and confidence score.
5. **Deployment** — The app is served using Streamlit for an interactive UI.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building & training |
| NumPy / Pandas | Data handling |
| Streamlit | Web app deployment |
| Jupyter Notebook | Experimentation & analysis |

---

## 💻 Installation

**1. Clone the repository**
```bash
git clone https://github.com/Goal48/IMDB_rating_using_Simple_RNN.git
cd IMDB_rating_using_Simple_RNN
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install tensorflow numpy pandas streamlit
```

---

## ▶️ Usage

**Run the Streamlit App:**
```bash
streamlit run main.py
```

Then open your browser at `http://localhost:8501`, type a movie review, and click **Predict Sentiment**.

**To retrain the model**, open and run `IMDB.ipynb` in Jupyter Notebook.

---

## 🏗️ Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Embedding → SimpleRNN → Dense |
| Vocabulary Size | 10,000 |
| Max Sequence Length | 500 |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Task | Binary Sentiment Classification |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | IMDB (50k reviews) |
| Threshold | 0.5 (Positive ≥ 0.5) |
| Model File | `imdb_rnn_model.h5` |

---

## 🔮 Future Improvements

- [ ] Upgrade to LSTM or GRU for better long-range dependency handling
- [ ] Add training accuracy/loss plots to the notebook
- [ ] Implement multi-class sentiment (Very Positive, Neutral, Very Negative)
- [ ] Deploy on Hugging Face Spaces or Streamlit Cloud
- [ ] Add a confidence score progress bar in the UI

---

## 👤 Author

**Goal48**
- GitHub: [@Goal48](https://github.com/Goal48)

