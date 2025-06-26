import os
from pathlib import Path

# File extensions to delete (customize as needed)
TARGET_EXTENSIONS = {".pdf", ".html", ".txt", ".xml", ".json", ".csv", ".zip", ".gz",".pkl",".pyc"}

# True = scans files to delete, False= deletes all files
def delete_files_in_directory(root_dir: str, dry_run: bool=True):
    """
    Deletes files with specified extensions in the given directory and subdirectories.

    :param root_dir: Path to the directory to clean.
    :param dry_run: If True, only logs what would be deleted.
    """
    deleted_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in TARGET_EXTENSIONS:
                if dry_run:
                    print(f"[Dry Run] Would delete: {file_path}")
                else:
                    try:
                        file_path.unlink()
                        print(f"Deleted: {file_path}")
                        deleted_files.append(str(file_path))
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

    return deleted_files

# Example usage
if __name__ == "__main__":
    # List of folders to clean — change these to your actual paths
    directories_to_clean = [
        "data/nasa",
        "data/pdfs",
        "data/wikipedia",
        "index",
        "src/__pycache__",
        "src/corpus_loader/__pycache__"
    ]

    dry_run = False  # Set to False to actually delete files

    total_deleted = 0
    for directory in directories_to_clean:
        print(f"\n--- Scanning: {directory} ---")
        deleted = delete_files_in_directory(directory, dry_run=dry_run)
        total_deleted += len(deleted)

    if dry_run:
        print(f"\n[Dry Run] Total files that would be deleted: {total_deleted}")
    else:
        print(f"\nTotal files deleted: {total_deleted}")
