from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "pcloud/job_catcher-bert-base-uncased"

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# Set model to evaluation mode
model.eval()

def predict(job_text, model, tokenizer):
    inputs = tokenizer(
        job_text,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()  # 0 = Real, 1 = Fake
    return "Fake Job Posting" if prediction == 1 else "Real Job Posting"

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
result = predict(new_job_posting, model, tokenizer)
print(result)
