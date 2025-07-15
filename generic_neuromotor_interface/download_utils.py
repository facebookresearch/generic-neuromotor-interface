# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url, output_path, description=None):
    """
    Download a file from a URL with a progress bar.

    Args:
        url (str): URL to download from
        output_path (Path): Path to save the file to
        description (str, optional): Description for the progress bar. Defaults to None.

    Returns:
        Path: Path to the downloaded file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get file size for progress bar
    total_size = int(response.headers.get("content-length", 0))
    desc = description or f"Downloading {output_path.name}"

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return output_path


def extract_tar(tar_path, extract_dir, description=None):
    """
    Extract a tar file with a progress bar.

    Args:
        tar_path (Path): Path to the tar file
        extract_dir (Path): Directory to extract to
        description (str, optional): Description for the progress bar. Defaults to None.

    Returns:
        Path: Path to the extraction directory
    """
    desc = description or f"Extracting {tar_path.name}"
    print(f"{desc}...")

    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting files"):
            tar.extract(member, path=extract_dir)

    return extract_dir


def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path (str or Path): Path to the directory

    Returns:
        Path: Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
