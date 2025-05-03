from openai import OpenAI
import os
from django.conf import settings

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="api_key"
)

def ai_summary(prompt):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": getattr(settings, 'SITE_URL', 'http://localhost:8000'),
                "X-Title": getattr(settings, 'SITE_NAME', 'Stat Analyzer')
            },
            model="meta-llama/llama-4-maverick:free",
            messages=[
                {"role": "system", "content": "You are a statistical analysis expert."},
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"