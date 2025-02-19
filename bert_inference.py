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
new_job_posting = "Alliance is looking for people who are passionate about both the work they do and the life they live.We are a high energy, goal oriented organization who is seeking out like minded professionals to assist our organization in continued growth. Alliance is an industry leader in providing innovative point-of-sale (POS) payment acceptance capabilities -- and backing them up with service, financial strength and stability.We offer flexibility, uncapped success, autonomy and life/work balance.You would be hard pressed to find a company that will treat you better. We offer daily/monthly contests and yearly bonuses for top employees. Come see why AMS is one of the fastest growing companies."
result = predict(new_job_posting, model, tokenizer)
print(result)
