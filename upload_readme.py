from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

api = HfApi()
api.upload_file(
    path_or_fileobj='dataset_card.md',
    path_in_repo='README.md',
    repo_id='justin-kelem/student-dropout-prediction',
    repo_type='dataset',
    token=os.getenv("HF_TOKEN")
)
print('README uploaded.')