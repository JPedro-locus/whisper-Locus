from dotenv import load_dotenv
import os

load_dotenv()  # carrega variáveis do .env

SILENCE_MIN_LENGTH = int(os.getenv("SILENCE_MIN_LENGTH", 100))
SILENCE_THRESH = int(os.getenv("SILENCE_THRESH", -35))
KEEP_SILENCE = int(os.getenv("KEEP_SILENCE", 50))
