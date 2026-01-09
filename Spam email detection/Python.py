import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("="*60)
print("SPAM EMAIL DETECTOR")
print("="*60)


print("\n1. Loading data...")
data = pd.read_csv('b_didebashvili25_84536_csv.csv')
print(f"   Loaded {len(data)} emails")

print("\n   Dataset columns:", list(data.columns))
print("\n   First 3 rows:")
print(data.head(3))

print("\n2. Preparing data...")

features = ['words', 'links', 'capital_words', 'spam_word_count']
X = data[features]  
y = data['is_spam']  

print(f"   Features used: {features}")
print(f"   Target column: is_spam")
print(f"   Feature matrix shape: {X.shape}")
print(f"   Spam emails: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
print(f"   Legitimate emails: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")


print("\n3. Training logistic regression model (70% of data)...")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Training on {X_train.shape[0]} emails")  
print(f"   Testing on {X_test.shape[0]} emails")    

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("   Model trained successfully!")

print("\n4. Model coefficients:")


coefficients = model.coef_[0]
intercept = model.intercept_[0]

coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

print("\n   Feature coefficients:")
print(coef_df.to_string(index=False))
print(f"\n   Model intercept: {intercept:.4f}")


coef_df.to_csv('logistic_regression_coefficients.csv', index=False)
print("   Saved to: logistic_regression_coefficients.csv")


print("\n5. Evaluating model on test data (30% of data)...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"   Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   [[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
print(f"    [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]")

print("\n6. Creating visualizations...")

os.makedirs('output', exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
spam_count = sum(y == 1)
legit_count = sum(y == 0)
bars = plt.bar(['Legitimate', 'Spam'], [legit_count, spam_count], 
               color=['green', 'red'], alpha=0.7)
plt.title('Email Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Email Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)


for bar, count in zip(bars, [legit_count, spam_count]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(count), ha='center', va='bottom', fontsize=11)

plt.subplot(1, 2, 2)
plt.pie([legit_count, spam_count], labels=['Legitimate', 'Spam'],
        autopct='%1.1f%%', colors=['green', 'red'], 
        startangle=90, explode=(0.05, 0.05))
plt.title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('output/class_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/class_distribution.png")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/confusion_matrix.png")

plt.figure(figsize=(10, 6))
colors = ['red' if coef > 0 else 'green' for coef in coef_df['coefficient']]
bars = plt.barh(range(len(coef_df)), coef_df['coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coef_df)), coef_df['feature'])
plt.title('Feature Importance for Spam Detection', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)


for bar, coef in zip(bars, coef_df['coefficient']):
    plt.text(coef + (0.01 if coef >= 0 else -0.01), 
             bar.get_y() + bar.get_height()/2,
             f'{coef:.3f}', 
             ha='left' if coef >= 0 else 'right',
             va='center',
             fontsize=10)

plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: output/feature_importance.png")


print("\n7. Email classification function:")

def check_email_features(words, links, capital_words, spam_word_count):
    """
    Classify an email based on its features
    """
    
    features = np.array([[words, links, capital_words, spam_word_count]])
    
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    result = "SPAM" if prediction == 1 else "LEGITIMATE"
    spam_prob = probability[1]
    
    return result, spam_prob


print("\n   Example 1 (Typical spam email):")
print("   words=200, links=8, capital_words=20, spam_word_count=6")
result1, prob1 = check_email_features(200, 8, 20, 6)
print(f"   Result: {result1}")
print(f"   Spam probability: {prob1:.2%}")


print("\n   Example 2 (Typical legitimate email):")
print("   words=180, links=1, capital_words=2, spam_word_count=0")
result2, prob2 = check_email_features(180, 1, 2, 0)
print(f"   Result: {result2}")
print(f"   Spam probability: {prob2:.2%}")


print("\n8. Creating example emails for report:")

my_spam_email = """
URGENT: WIN $1,000,000 PRIZE!
Dear Winner,

Congratulations! You have been selected to win $1,000,000 CASH PRIZE!
This is a LIMITED TIME OFFER. Click here to claim: http://bit.ly/winprize123

To claim your prize, we need $99 processing fee.
This is 100% GUARANTEED! Don't miss this opportunity!

Reply within 24 hours.
"""

print(f"\n   My Spam Email:")
print("   " + my_spam_email[:100] + "...")
print("   Features: words=85, links=1, capital_words=7, spam_word_count=9")
spam_result, spam_prob = check_email_features(85, 1, 7, 9)
print(f"   Result: {spam_result} (Probability: {spam_prob:.2%})")

my_legit_email = """
Subject: Project Update Meeting

Hello Team,

The project update meeting is scheduled for tomorrow at 2 PM in Conference Room A.
Please bring your progress reports and be ready to discuss next week's milestones.

The presentation slides are attached to this email.
Looking forward to our discussion.

Best regards,
Alex Johnson
Project Lead
"""

print(f"\n   My Legitimate Email:")
print("   " + my_legit_email[:100] + "...")
print("   Features: words=65, links=0, capital_words=3, spam_word_count=0")
legit_result, legit_prob = check_email_features(65, 0, 3, 0)
print(f"   Result: {legit_result} (Probability: {legit_prob:.2%})")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

TN, FP, FN, TP = cm.ravel()

print(f"""
📊 DATASET:
   - Total emails: {len(data)}
   - Spam: {spam_count} ({spam_count/len(data)*100:.1f}%)
   - Legitimate: {legit_count} ({legit_count/len(data)*100:.1f}%)

🤖 MODEL:
   - Algorithm: Logistic Regression
   - Training samples: {X_train.shape[0]} (70%)
   - Testing samples: {X_test.shape[0]} (30%)

📈 PERFORMANCE:
   - Accuracy: {accuracy*100:.2f}%
   - True Positives: {TP}
   - True Negatives: {TN}
   - False Positives: {FP}
   - False Negatives: {FN}

🔑 FEATURE IMPORTANCE:
   {coef_df.to_string(index=False)}

📁 FILES CREATED:
   - logistic_regression_coefficients.csv
   - output/class_distribution.png
   - output/confusion_matrix.png
   - output/feature_importance.png

📧 TO TEST NEW EMAIL:
   result, prob = check_email_features(words, links, capital_words, spam_word_count)
""")

plt.show()