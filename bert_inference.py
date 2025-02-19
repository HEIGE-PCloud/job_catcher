from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import util
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate(y_test, y_pred):
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  print(f'Confusion matrix:\n {confusion_matrix(y_test, y_pred)}')
  print(f"{classification_report(y_test, y_pred)}")

model_name = "pcloud/job_catcher-bert-base-uncased"
MAX_LENGTH = 512

# Set device to "mps" if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)  # Move the model to the device
model.eval()

def predict(job_text, model, tokenizer, device):
    inputs = tokenizer(
        job_text,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)  # Move input tensors to the device
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()  # 0 = Real, 1 = Fake
    return prediction

# Example usage:
new_job_posting = """
**Job Posting: Work-From-Home Data Entry Specialist**

**Company:** Global Solutions Inc.

**Location:** Remote

**Salary:** $50,000 - $70,000 per year + bonuses

**Job Type:** Full-Time

**About Us:**
Global Solutions Inc. is a leading provider of innovative services across various industries. We pride ourselves on our commitment to excellence and our dynamic team of professionals. As we expand our operations, we are seeking motivated individuals to join us in our mission to deliver outstanding results for our clients.

**Job Description:**
We are looking for a detail-oriented Work-From-Home Data Entry Specialist to join our team. In this role, you will be responsible for inputting, updating, and maintaining data in our systems. This is a fantastic opportunity for individuals looking to work from the comfort of their own home.

**Key Responsibilities:**
- Enter and update data accurately in our database.
- Review and verify data entries for accuracy.
- Assist in generating reports as needed.
- Communicate effectively with team members and management.
- Participate in virtual meetings and training sessions.

**Qualifications:**
- High school diploma or equivalent; college degree preferred.
- Strong attention to detail and organizational skills.
- Proficiency in Microsoft Office Suite and data entry software.
- Excellent communication skills.
- Ability to work independently and manage time effectively.

**What We Offer:**
- Flexible work hours.
- Competitive salary with performance-based bonuses.
- Comprehensive training and ongoing support.
- Opportunity for career growth and advancement.
- Work in a thriving, remote-first environment.

**How to Apply:**
If you are interested in joining our team, please submit your resume and a cover letter detailing your relevant experience to [email@example.com]. In your email, please also include your full name, phone number, and address for verification purposes.

**Important Note:**
Upon selection, you will be required to complete a background verification process. Additionally, please be prepared to purchase a starter software package for your home office setup, which will be reimbursed within your first month of employment.

**Global Solutions Inc. is an equal opportunity employer. We celebrate diversity and are committed to creating an inclusive environment for all employees.**

*Disclaimer: This job posting is intended for informational purposes only. No financial information will be requested during the application process.* 

**Apply now and take the first step toward a rewarding career with Global Solutions Inc.!**
"""
result = predict(new_job_posting, model, tokenizer, device)
print(f"Prediction: {'Fake' if result == 1 else 'Real'}")

df = util.load_data(Path("data/fake_job_postings.csv"))
X_train, X_test, y_train, y_test = util.split_data(df)

# Get predictions for the test set using the device
predictions = []
for text in tqdm(X_test):
    prediction = predict(text, model, tokenizer, device)
    predictions.append(prediction)

evaluate(y_test, predictions)
# Save false positive and false negative job postings to files.

# False positive: predicted Fake (1) but actually Real (0)
# False negative: predicted Real (0) but actually Fake (1)
false_positives = [
    text for text, truth, pred in zip(X_test, y_test, predictions)
    if truth == 0 and pred == 1
]
false_negatives = [
    text for text, truth, pred in zip(X_test, y_test, predictions)
    if truth == 1 and pred == 0
]

with open("false_positives.txt", "w") as fp_file:
    for job in false_positives:
        fp_file.write(job + "\n ================= \n")

with open("false_negatives.txt", "w") as fn_file:
    for job in false_negatives:
        fn_file.write(job + "\n ================= \n")

print(f"Saved {len(false_positives)} false positives in false_positives.txt")
print(f"Saved {len(false_negatives)} false negatives in false_negatives.txt")

# Accuracy: 0.9851373182552504
# Confusion matrix:
#  [[2943   15]
#  [  31  106]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      2958
#            1       0.88      0.77      0.82       137

#     accuracy                           0.99      3095
#    macro avg       0.93      0.88      0.91      3095
# weighted avg       0.98      0.99      0.98      3095
