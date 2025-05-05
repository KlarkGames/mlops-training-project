import os
from dotenv import load_dotenv

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()

competition = "asr-numbers-recognition-in-russian"
output_dir = os.getenv("INITIAL_DATA_PATH", "data")

api.competition_download_files(competition=competition, path=output_dir)

zip_path = os.path.join(output_dir, f"{competition}.zip")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(output_dir)

print(f"Data ready in: {output_dir}")
