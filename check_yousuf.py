import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src module can be found
sys.path.append(os.path.abspath(''))

from src.utils import load_data
from src.preprocessing import clean_text, create_vectorizer
from src.model import train_model, predict_email

# Load Data
DATA_PATH = "spam_ham_dataset.csv/spam_ham_dataset.csv"
try:
    df = load_data(DATA_PATH)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Preprocessing
df['clean_text'] = df['text'].apply(clean_text)
vectorizer = create_vectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']

# Train Model (Full data for best result)
model = train_model(X, y)

# User's Email
user_email = """Hi there,

I hope you’re doing well. I just wanted to reach out and introduce myself. My name is Yousuf, and I’m currently exploring new opportunities to collaborate on interesting and impactful projects.

I’m particularly interested in innovative ideas that combine technology and real-world problem solving. If there are any upcoming projects, discussions, or opportunities where I could contribute, I would be more than happy to connect and learn more.

Please feel free to let me know a convenient time for a quick conversation. I look forward to hearing from you.

Best regards,
Yousuf"""

# Prediction
pred = predict_email(model, vectorizer, user_email, clean_text)
print(f"Prediction for Yousuf's email: {pred}")
