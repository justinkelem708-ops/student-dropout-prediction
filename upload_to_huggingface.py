from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import os
load_dotenv('agent/.env')

# Ton token Hugging Face
token=os.getenv("HF_TOKEN")
USERNAME = "justin-kelem"
REPO_NAME = "student-dropout-prediction"

# Créer le repo dataset
api = HfApi()

create_repo(
    repo_id=f"{USERNAME}/{REPO_NAME}",
    token=TOKEN,
    repo_type="dataset",
    exist_ok=True
)

print("Repository created.")

# Uploader les fichiers
files_to_upload = [
    ("data/X_train.csv", "X_train.csv"),
    ("data/X_test.csv", "X_test.csv"),
    ("data/y_train.csv", "y_train.csv"),
    ("data/y_test.csv", "y_test.csv"),
    ("data/student-mat.csv", "student-mat.csv"),
    ("data/student-por.csv", "student-por.csv"),
]

for local_path, repo_path in files_to_upload:
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=f"{USERNAME}/{REPO_NAME}",
            repo_type="dataset",
            token=TOKEN
        )
        print(f"Uploaded : {repo_path}")
    else:
        print(f"Not found : {local_path}")

print("Upload complete.")