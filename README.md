# 📰 Fake News Detection Using Machine Learning

## 📖 Overview

This project is a Machine Learning-based Fake News Detection System that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and a **Passive Aggressive Classifier**.

The system preprocesses news text, converts it into numerical features using **TF-IDF Vectorization**, and predicts whether the news article is genuine or misleading.

---

## 🎯 Project Goals

* Detect fake news articles automatically.
* Reduce the spread of misinformation.
* Apply NLP techniques for text processing.
* Build an efficient and lightweight machine learning classifier.

---

## ✨ Features

* News authenticity prediction (Real/Fake)
* Text preprocessing and cleaning
* Dataset shuffling and preparation
* TF-IDF feature extraction
* Passive Aggressive Classifier implementation
* Accuracy and confusion matrix evaluation
* Easy-to-understand code structure

---

## 🛠 Tech Stack

### Programming Language

* Python 3

### Libraries

* Pandas
* Regular Expressions (re)
* Scikit-learn

### Machine Learning Components

* TF-IDF Vectorizer
* Passive Aggressive Classifier
* Train-Test Split
* Accuracy Evaluation
* Confusion Matrix

---

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── fake_news_detection.py
│
├── requirements.txt
│
└── README.md
```

---

## 🗂 Dataset

The project uses two datasets:

### Fake.csv

Contains fake news articles.

### True.csv

Contains genuine news articles.

After loading:

* Fake News → Label = 0
* Real News → Label = 1

The datasets are merged and shuffled before training.

---

## ⚙️ Working Process

### 1. Data Loading

Both datasets are loaded using Pandas.

```python
df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')
```

### 2. Label Assignment

```python
df_fake['label'] = 0
df_true['label'] = 1
```

### 3. Data Cleaning

The text is cleaned by:

* Removing HTML tags
* Removing special characters
* Converting text to lowercase

Example:

```text
Before:
"<h1>Breaking News!!!</h1>"

After:
"breaking news"
```

### 4. Feature Extraction

TF-IDF Vectorization converts text into numerical vectors.

```python
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)
```

### 5. Model Training

A Passive Aggressive Classifier is trained using the transformed data.

```python
model = PassiveAggressiveClassifier(max_iter=50)
```

### 6. Prediction

The trained model predicts whether the news is Real or Fake.

---

## 🧠 Why Passive Aggressive Classifier?

The Passive Aggressive algorithm is:

* Fast
* Memory efficient
* Suitable for text classification
* Effective on large datasets
* Commonly used in fake news detection tasks

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Rajnish09G/Fake-News-Detection.git
cd Fake-News-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python fake_news_detection.py
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

### Accuracy Score

```python
accuracy_score(y_test, y_pred)
```

### Confusion Matrix

```python
confusion_matrix(y_test, y_pred)
```

Example Output:

```text
Accuracy: 99.20%
[[6289   42]
 [  66 7073]]
```

(Note: Actual values may vary.)

---

## 🔍 Example Prediction

Input:

```text
U.S. military to accept transgender recruits on Monday
```

Output:

```text
Real
```

---

## 📈 Machine Learning Workflow

```text
News Article
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Passive Aggressive Classifier
      │
      ▼
Real / Fake Prediction
```

---

## 🔮 Future Improvements

* Deploy using Flask or Django
* Build a Web Application
* Create a REST API
* Use Deep Learning Models (LSTM)
* Implement BERT-based Classification
* Add Real-Time News Verification
* Create Browser Extension Support

---

## 📚 Learning Outcomes

Through this project:

* Applied NLP preprocessing techniques
* Learned text vectorization using TF-IDF
* Implemented a Machine Learning classification model
* Evaluated model performance using accuracy and confusion matrix
* Understood real-world fake news detection systems

---

## 🤝 Contributors

* Rajnish09G
* Prem2905

Feel free to contribute by creating pull requests.

---


## ⭐ Support

If you found this project helpful:

⭐ Star this repository

🍴 Fork this repository

🛠 Contribute to improvements

---

### Made with Python, Machine Learning, and NLP
