import openai

client = openai.OpenAI()

if __name__ == "__main__":
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming.",
            },
        ],
    )

    print(response.choices[0].message)
