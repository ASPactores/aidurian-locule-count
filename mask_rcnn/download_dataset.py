import os
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    project_version: str,
    dataset_format: str,
    home_dir: str,
    dataset_path: str,
) -> None:
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        os.chdir(dataset_path)

        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project_name)
            version = project.version(project_version)
            version.download(dataset_format)
            print(f"Dataset downloaded successfully in {dataset_path}")
        except Exception as e:
            print(f"Error downloading the dataset: {e}")
        finally:
            os.chdir(home_dir)
    else:
        print(f"Dataset directory already exists at {dataset_path}. Skipping download.")
    os.chdir(home_dir)


if __name__ == "__main__":
    # Environment variables
    api_key = os.getenv("API_KEY", "your_default_api_key")
    workspace = os.getenv("WORKSPACE", "your_workspace_name")
    project_name = os.getenv("PROJECT", "your_project_name")
    project_version = os.getenv("PROJECT_VERSION", "1")
    dataset_format = os.getenv("DATASET_FORMAT", "your_dataset_format")

    home_dir = os.getcwd()
    dataset_path = os.path.join(home_dir, "dataset")

    download_dataset()
