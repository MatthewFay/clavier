import os
import re

import requests

#COMPOSER: str = "bach"
# BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/jsbachchorales.abc"

# COMPOSER: str = "mozart"
# BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/mozart_sonatas.abc"

# COMPOSER: str = "scarlatti"
# BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/scarlatti_sonatas.abc"

# COMPOSER: str = "beethoven"
# # BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/beethoven_sonatas.abc"
# BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/beethoven_quartets.abc"

# COMPOSER:str = "chopin"
# # BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/chopin_1stEdition.abc"
# # BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/chopinpreludes.abc"
# BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/chopinmazurkas.abc"

COMPOSER: str = "haydn"
BULK_URL: str = "https://ifdo.ca/~seymour/kern2abc/haydn_sonatas.abc"

# 1. Path Fix: Dynamically resolve the project root
# so it always writes to the right place
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "data", "raw", COMPOSER)

def sanitize_filename(text: str) -> str:
    """Convert a title into a filesystem-safe string."""
    text = text.lower().replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading collection from: {BULK_URL}")
    response = requests.get(BULK_URL)
    response.raise_for_status()
    
    raw_text: str = response.text
    
    # 2. Junk Fix: Split using regex to only match 'X:' at the start of a line
    chunks: list[str] = re.split(r"^X:", raw_text, flags=re.MULTILINE)
    
    tunes: list[str] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        # If the chunk lacks a Title tag (T:), it's the global file header (metadata).
        # Skip it.
        if not re.search(r"^T:", chunk, flags=re.MULTILINE):
            continue
            
        tunes.append("X:" + chunk)
    
    print(f"Successfully downloaded {len(tunes)} tunes. Splitting into files...")
    
    saved_count: int = 0
    for tune in tunes:
        # Extract the title (T:) for the filename
        title_match = re.search(r"^T:(.+)$", tune, re.MULTILINE)
        raw_title: str = title_match.group(1).strip() if title_match else "unknown"
        clean_title: str = sanitize_filename(raw_title)
        
        # Extract the index (X:) for the filename
        index_match = re.search(r"^X:\s*(\d+)", tune, re.MULTILINE)
        tune_index: int = int(index_match.group(1)) if index_match else saved_count + 1
        
        filename: str = f"{COMPOSER}_{tune_index:04d}_{clean_title}.abc"
        filepath: str = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(tune.strip() + "\n")
            
        saved_count += 1
        
    print(f"Done! Extracted {saved_count} clean {COMPOSER} files to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()