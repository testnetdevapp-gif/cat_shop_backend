import os
from dotenv import load_dotenv

load_dotenv()  # โหลด .env ก่อน

from google import genai  # ใช้ library ใหม่

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY_DETECT"])
for m in client.models.list():
    print(m.name)
 