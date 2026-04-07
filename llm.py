import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, context):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided context. Do not chat or give suggestions."
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {query}

                Answer in 3-5 sentences:
                """
            }
        ]
    )
    return response.choices[0].message.content