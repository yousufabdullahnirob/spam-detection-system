# 📧 Spam Email Detection using Naive Bayes & TF-IDF

A machine learning project that classifies emails as **Spam** or **Ham (Not Spam)** using classical Natural Language Processing techniques.

This project demonstrates how TF-IDF vectorization combined with a Multinomial Naive Bayes classifier can effectively solve text classification problems with high accuracy and efficiency.

---

## 🎯 Project Highlights

- Clean and modular project structure  
- Text preprocessing pipeline  
- TF-IDF feature engineering  
- Multinomial Naive Bayes model  
- Model evaluation with accuracy & detailed metrics  
- Interactive real-time prediction  

---

## 🧠 How It Works

1. **Preprocessing**  
   - Convert text to lowercase  
   - Remove punctuation and numbers  
   - Clean unnecessary whitespace  

2. **Vectorization**  
   - Transform text into numerical features using TF-IDF  

3. **Model Training**  
   - Train a Multinomial Naive Bayes classifier  

4. **Evaluation**  
   - Accuracy  
   - Confusion Matrix  
   - Precision, Recall, F1-score  

5. **Prediction**  
   - Classify custom input messages directly from the console  

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  

---

## 📂 Project Structure

├── main.py
├── spam_ham_dataset.csv/
├── src/
│ ├── model.py
│ ├── preprocessing.py
│ ├── utils.py
│ └── visualization.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
pip install -r requirements.txt
▶️ Run the Project
python main.py
After running:

The model will train on the dataset

Evaluation metrics will be displayed

Interactive mode will allow real-time spam detection

📊 Example Output
Accuracy: 0.97+

Enter email text:
"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."

Prediction: Spam
🚀 Why This Project?
Spam detection is a real-world application of NLP and machine learning used in email services, messaging platforms, and cybersecurity systems.

This project focuses on building a strong foundation in:

Text preprocessing

Feature engineering

Probabilistic classification

Model evaluation
