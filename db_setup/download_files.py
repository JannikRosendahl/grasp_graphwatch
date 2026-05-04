import gdown
import ast
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load .env file from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")


def load_drive_links_from_env():
    keys = [
        "optc_and_cadets_theia_clearscope_e3",
        "theia_clearscope_e5",
        "cadets_e5",
    ]
    return [os.getenv(key) for key in keys if os.getenv(key)]


drive_links = load_drive_links_from_env() or []


def extract_file_id(url):
    # Extracts the file ID from a Google Drive URL
    parts = url.split("/")
    try:
        file_id_index = parts.index("d") + 1
        return parts[file_id_index]
    except (ValueError, IndexError):
        return None


def download_files(links):
    for url in links:
        logger.info(f"Downloading file from URL: {url}")
        file_id = extract_file_id(url)
        if file_id:
            gdown.download(id=file_id, output=None, quiet=False)
        else:
            print(f"Could not extract file ID from URL: {url}")


if __name__ == "__main__":
    download_files(drive_links)
