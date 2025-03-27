# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# TODO: delete before OSS
import os

COPYRIGHT_HEADER = """# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree."""

PYRE_HEADER = "# pyre-strict"

HEADER_WITH_PYRE = PYRE_HEADER + "\n" + COPYRIGHT_HEADER


def add_header_to(directory: str) -> None:
    """
    Recursively adds a HEADER_WITH_PYRE to all .py files in the given directory.

    Args:
        directory (str): The root directory to start from.
    """

    # Walk through the directory tree and find all .py files
    all_py_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                all_py_files.append(filepath)

    # Ask for confirmation before modifying any files
    print(f"Found {len(all_py_files)} .py files in the directory tree.")
    directory_base_name = os.path.basename(directory)
    for py_file in all_py_files:
        print(os.path.join(directory_base_name, os.path.relpath(py_file, directory)))

    print(f"{directory=}")
    if (
        input("Do you want to add the header to all these files? (y/n): ").lower()
        != "y"
    ):
        print("Aborting.")
        return

    # Add the HEADER_WITH_PYRE to each file
    for filepath in all_py_files:

        if filepath == __file__:
            continue

        # Read the existing content of the file
        with open(filepath, "r") as file:
            content = file.read()

        # Remove any existing headers
        content = content.replace(PYRE_HEADER, "").replace(COPYRIGHT_HEADER, "")

        # Add the new header
        content = HEADER_WITH_PYRE + "\n\n" + content.lstrip()

        # Write the updated content back to the file
        with open(filepath, "w") as file:
            file.write(content)


def check_all_files_have_header(directory: str) -> None:
    """
    Recursively checks that all .py files in the given directory have the correct header.

    Args:
        directory (str): The root directory to start from.
    """

    # Walk through the directory tree and find all .py files
    all_py_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                all_py_files.append(filepath)

    # Check each file
    errors = False
    for filepath in all_py_files:

        if filepath == __file__:
            continue

        with open(filepath, "r") as file:
            content = file.read()

        # Check for duplicate headers
        if content.count(PYRE_HEADER) > 1 or content.count(COPYRIGHT_HEADER) > 1:
            print(f"Error: Duplicate headers found in {filepath}")
            errors = True

        # Check for missing headers
        if not content.startswith(HEADER_WITH_PYRE):
            print(f"Error: Missing header in {filepath}")
            errors = True

    if errors:
        raise ValueError("Errors found. Please fix them and run the script again.")
    else:
        print("All files passed sanity checks. Please sanity check manually.")


if __name__ == "__main__":
    # Get the parent-parent-directory of the current script (generic_neuromotor_interface/)
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    add_header_to(directory)
    check_all_files_have_header(directory)
