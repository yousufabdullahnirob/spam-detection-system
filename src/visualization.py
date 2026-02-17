import matplotlib.pyplot as plt
import seaborn as sns

def plot_spam_ham_count(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title("Spam vs Ham Distribution")
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_text_length_distribution(df):
    # Check if length column exists, if not create it temporarily or assume it exists. 
    # Better to calculate it here to be safe or ensure it's passed with it.
    if 'length' not in df.columns:
        df = df.copy()
        df['length'] = df['text'].apply(len)
        
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='length', hue='label', kde=True, bins=50)
    plt.title("Email Length Distribution")
    plt.show()
