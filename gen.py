import openai
import pandas as pd
from tqdm import tqdm


client = openai.OpenAI()

def create_fake_job_posting():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cyber security expert who knows a lot about the fraudulent job postings."},
            {
                "role": "user",
                "content": "Write a realistic job posting that's actually a scam, trying to trick people into giving away their personal information and their money.",
            },
        ],
        temperature=0.8
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    job_descriptions = []
    for _ in tqdm(range(1)):
        job_descriptions.append(create_fake_job_posting())

    df = pd.DataFrame({
        'job_description': job_descriptions,
        'fraudulent': 1
    })

    # Try to read existing CSV file, if it exists
    try:
        existing_df = pd.read_csv('data/gpt-4o-mini-fake_job_postings.csv')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv('data/gpt-4o-mini-fake_job_postings.csv', index=False)
