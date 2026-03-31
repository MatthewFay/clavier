import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

# Define the Special Tokens
BOS_TOKEN: str = "<|bos|>"
EOS_TOKEN: str = "<|eos|>"


def clean_abc_text(raw_abc: str, composer: str) -> str:
    """
    Strips noise, inline comments, layout instructions, backslashes,
    converts global headers to inline brackets, removes whitespace, 
    applies gatekeeper checks, and packs the sequence tightly with special tokens.
    """
    lines: list[str] = raw_abc.splitlines()

    headers: list[str] = []
    music_parts: list[str] = []

    for line in lines:
        # 1. Strip inline comments (keep only what's to the left of '%')
        if "%" in line:
            line = line.split("%")[0]

        line = line.strip()

        # Skip empty lines after comment removal
        if not line:
            continue

        # 2. Process Headers
        if len(line) >= 2 and line[1] == ":" and line[0].isalpha():
            header_type: str = line[0].upper()

            # Keep only the essential mathematical headers
            if header_type in ["M", "L", "K", "V"]:
                clean_header: str = line.replace(" ", "")
                headers.append(f"[{clean_header}]")

        # 3. Process Music
        else:
            clean_music: str = line.replace(" ", "")

            # Strip out layout instructions like [I:setbarnb1]
            # Wrapped in str() to satisfy Pylance strict mode
            clean_music = str(re.sub(r"\[I:[^\]]*\]", "", clean_music))

            # Strip out line continuation backslashes
            clean_music = clean_music.replace("\\", "")

            # If the line is empty after stripping, skip it
            if not clean_music:
                continue

            music_parts.append(clean_music)

    # --- GATEKEEPER CHECKS ---
    
    # 1. The Ghost Check: Did we actually extract any music?
    if not music_parts:
        return ""
        
    # 2. The Monster Check: Count the voices.
    joined_headers: str = "".join(headers)
    voice_matches: list[str] = re.findall(r"\[V:(\d+)", joined_headers)
    if voice_matches:
        max_voice: int = max(int(v) for v in voice_matches)
        if max_voice > 4:
            return ""  # Drop this file, it's too complex for the V1 model
            
    # 3. The Length Check: Prevent Context Window Blowouts
    joined_music: str = "".join(music_parts)
    if len(joined_music) > 3000:
        return ""
        
    # --- END GATEKEEPER CHECKS ---

    # 4. Compile the Sequence
    composer_token: str = f"<|{composer.lower()}|>"

    final_sequence: str = (
        BOS_TOKEN
        + composer_token
        + joined_headers
        + joined_music
        + EOS_TOKEN
    )

    return final_sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and compile ABC files into a tightly packed JSONL dataset."
    )
    parser.add_argument(
        "--composer",
        type=str,
        required=True,
        help="Name of the composer (e.g., bach, mozart)",
    )
    parser.add_argument(
        "--input-dir", type=str, help="Optional specific input directory."
    )
    args = parser.parse_args()

    composer: str = args.composer.lower()

    # Path resolution
    script_dir: Path = Path(__file__).parent.resolve()
    project_root: Path = script_dir.parent

    input_dir: Path = (
        Path(args.input_dir)
        if args.input_dir
        else project_root / "data" / "raw" / composer
    )
    output_dir: Path = project_root / "data" / "processed" / composer
    output_file: Path = output_dir / f"{composer}.jsonl"

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    abc_files: list[Path] = list(input_dir.rglob("*.abc"))

    if not abc_files:
        print(f"No .abc files found in {input_dir}")
        return

    print(f"Found {len(abc_files)} files for {composer}.")
    print(f"Compiling tightly packed sequences to {output_file}...")

    success_count: int = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in tqdm(abc_files, desc="Processing"):
            try:
                with open(file_path, encoding="utf-8") as infile:
                    raw_text: str = infile.read()

                cleaned_text: str = clean_abc_text(raw_text, composer)

                # Skip files that failed the gatekeeper checks
                if not cleaned_text:
                    continue

                record: dict[str, str] = {"composer": composer, "text": cleaned_text}

                outfile.write(json.dumps(record) + "\n")
                success_count += 1

            except Exception as e:
                tqdm.write(f"Failed to process {file_path.name}: {e}")

    print("\n" + "=" * 50)
    print("Dataset compilation complete!")
    print(f"Successfully processed {success_count} / {len(abc_files)} files.")
    print(f"Output: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()