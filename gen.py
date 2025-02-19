import openai

client = openai.OpenAI()

if __name__ == "__main__":
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

    print(response.choices[0].message.content)
