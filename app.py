import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load datasets
df_fake = pd.read_csv('dataset/Fake.csv', engine='python', on_bad_lines='skip')
df_true = pd.read_csv('dataset/True.csv', engine='python', on_bad_lines='skip')

# Assign labels
df_fake['label'] = 0  # Fake
df_true['label'] = 1  # Real

# Combine data
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Use only necessary columns
df = df[['text', 'label']]

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')
print(confusion_matrix(y_test, y_pred))

def predict_news(news):
    news_clean = clean_text(news)
    news_tfidf = vectorizer.transform([news_clean])
    pred = model.predict(news_tfidf)
    return "Real" if pred[0] == 1 else "Fake"

# Example usage:
print(predict_news("U.S. military to accept transgender recruits on Monday"))