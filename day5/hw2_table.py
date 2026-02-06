import pandas as pd
from openai import OpenAI

tables = pd.read_html("table_html.html")
# print(tables[0]) # Screenshot doesn't show this print, but user had it. I'll comment it out or keep it? User said "add screenshot code". I'll keep previous code but maybe comment print to avoid clutter if screenshot implies clean start? Actually, screenshot uses tables[0], so tables must be defined.

# Using Prompt_table_v2.txt as v1 is missing
with open("Prompt_table_v2.txt", "r", encoding="UTF-8") as f:
    system_prompt = f.read()

client = OpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="/models/gpt-oss-120b",
    messages=[
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{tables[0].to_string()}"}
    ],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    },
    stream=True
)

# 逐步接收回應
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)