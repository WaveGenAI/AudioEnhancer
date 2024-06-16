""" 
This script downloads the dataset from the MTG server.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

# Constants
CHUNK_SIZE = 512 * 1024
BASE_URL = "https://cdn.freesound.org/mtg-jamendo/"


def download_from_mtg_fast(download_file: str, download_path: str):
    """Download a file from the MTG server.

    Args:
        download_file (str): the file to download
        download_path (str): the path to save the file
    """
    res = requests.get(
        BASE_URL + f"raw_30s/audio/{download_file}", stream=True, timeout=10
    )

    total = res.headers.get("Content-Length")
    if total is not None:
        total = int(total)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file_d:
        with tqdm(total=total, unit="B", unit_scale=True) as progressbar:
            for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
                tmp_file_d.write(chunk)
                progressbar.update(len(chunk))
        shutil.move(tmp_file_d.name, os.path.join(download_path, download_file))


def download(download_dir: str, nb_files: int = 1000):
    """Download the dataset from the MTG server.

    Args:
        download_dir (str): the directory to save the files
        nb_files (int, optional): the number of tar files to download. Defaults to 1000.

    Raises:
        ValueError: if the hash of one of the files is incorrect
    """

    file_sha256_tars = BASE_URL + "raw_30s/audio/checksums_sha256.txt"

    # download the checksums file
    res = requests.get(file_sha256_tars, timeout=10)
    res.raise_for_status()

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    checksums = res.text.split("\n")

    for i, checksum in enumerate(checksums):
        try:
            hash_tar_file, file = checksum.split(" ")
        except ValueError:
            continue

        print(f"Downloading {file}")
        download_from_mtg_fast(file, download_dir)

        # check if the hash is correct
        with open(os.path.join(download_dir, file), "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != hash_tar_file:
                raise ValueError(f"Hash mismatch for {file}")

        if (i + 1) >= nb_files:
            break

    print("Download complete.")

    # extract the files

    for file in os.listdir(download_dir):
        if file.endswith(".tar"):
            print(f"Extracting {file}")
            path = os.path.join(download_dir, file)
            shutil.unpack_archive(path, download_dir)
            os.remove(path)

    # move all the file in directory to the download_dir
    for directory in os.listdir(download_dir):
        dir_path = os.path.join(download_dir, directory)
        for file in os.listdir(dir_path):
            shutil.move(os.path.join(dir_path, file), download_dir)
        os.rmdir(dir_path)

    print("Extraction complete.")
