# Copy into one Colab cell to download both GPU experiments and diagnostics.
from pathlib import Path
import zipfile
from google.colab import files

base = Path("/content/drive/MyDrive/TrialGPT_Assignment2")
folders = [base / "kv_cache_v1", base / "kv_cache_math_v1"]
for folder in folders:
    if not folder.is_dir():
        raise FileNotFoundError(f"Results folder not found: {folder}")

download = Path("/content/kv_cache_results_all.zip")
with zipfile.ZipFile(download, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file():
                name = Path(folder.name) / path.relative_to(folder)
                archive.write(path, name.as_posix())

print("Saved both benchmark runs, including the available numerical diagnostic.")
files.download(str(download))
