import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src module can be found
sys.path.append(os.path.abspath(''))

from src.utils import load_data
from src.preprocessing import clean_text, create_vectorizer
from src.model import train_model, evaluate_model, predict_email

def main():
    print("=========================================")
    print("   Spam Detection Pipeline Started       ")
    print("=========================================")
    
    #  Load Data
    DATA_PATH = "spam_ham_dataset.csv/spam_ham_dataset.csv"
    print(f"[INFO] Loading data from {DATA_PATH}...")
    try:
        df = load_data(DATA_PATH)
        print(f"[SUCCESS] Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # 2. Preprocessing
    print("[INFO] Preprocessing text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("[INFO] Vectorizing text (TF-IDF)...")
    vectorizer = create_vectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label_num']
    print(f"[SUCCESS] Vectorization complete. Features: {X.shape[1]}")
    
    # 3. Train/Test Split
    print("[INFO] Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"       Train size: {X_train.shape[0]}")
    print(f"       Test size:  {X_test.shape[0]}")
    
    # 4. Model Training
    print("[INFO] Training Naive Bayes model...")
    model = train_model(X_train, y_train)
    print("[SUCCESS] Model trained.")
    
    # 5. Evaluation
    print("[INFO] Evaluating model...")
    acc, report, cm, _ = evaluate_model(model, X_test, y_test)
    print(f"-----------------------------------------")
    print(f" Accuracy: {acc:.4f}")
    print(f"-----------------------------------------")
    print(" Classification Report:")
    print(report)
    print("-----------------------------------------")
    print(" Confusion Matrix:")
    print(cm)
    print("-----------------------------------------")
    
    # 6. Custom Prediction
    print("[INFO] Running custom predictions...")
    samples = [
        "Congratulations! You won 1 million dollars",
        "Hey, can we meet for lunch tomorrow?",
        "URGENT: Your account has been compromised. Click here to reset password.",
        """Hi there,

I hope you’re doing well. I just wanted to reach out and introduce myself. My name is Yousuf, and I’m currently exploring new opportunities to collaborate on interesting and impactful projects.

I’m particularly interested in innovative ideas that combine technology and real-world problem solving. If there are any upcoming projects, discussions, or opportunities where I could contribute, I would be more than happy to connect and learn more.

Please feel free to let me know a convenient time for a quick conversation. I look forward to hearing from you.

Best regards,
Yousuf""",
        "Free money!!!"
    ]
    
    for text in samples:
        pred = predict_email(model, vectorizer, text, clean_text)
        print(f"   Input: '{text}'")
        print(f"   Prediction: {pred}")
        print("   ---")
    
    print("=========================================")
    print("   Interactive Mode                      ")
    print("=========================================")
    print("Type your email text below to check if it's Spam or Ham.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nEnter email text: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        pred = predict_email(model, vectorizer, user_input, clean_text)
        print(f"Prediction: {pred}")
    
    print("=========================================")
    print("   Pipeline Finished Successfully        ")
    print("=========================================")

if __name__ == "__main__":
    main()
