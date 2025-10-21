from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging

# Logging setup
log_format = "%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Initialize LLM
try:
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    logger.info("ChatGroq model initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize ChatGroq", exc_info=True)

# Example invocation
# response = llm.invoke("hey can you build me a calculator app in python?")
# logger.info("Received response from model")
# print(response.content)
