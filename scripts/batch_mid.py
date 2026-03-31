import subprocess
from pathlib import Path
from typing import Any, cast

# Import music21 for our fallback parser
from music21 import converter
from tqdm import tqdm

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PROJECT_ROOT: Path = SCRIPT_DIR.parent

# Separate Input and Output directories
INPUT_DIR: Path = PROJECT_ROOT / "data" / "raw_mid" / "bach"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "raw" / "bach"


def convert_midi_to_abc(midi_path: Path) -> bool:
    """
    Converts a single MIDI file to ABC notation.
    Uses a Python fallback for corrupted files.
    """
    # Route the output to the new data/raw/bach directory
    abc_path: Path = OUTPUT_DIR / f"{midi_path.stem}.abc"

    if abc_path.exists():
        return True

    command: list[str] = ["midi2abc", str(midi_path), "-o", str(abc_path)]

    try:
        # 1. The Fast Path (C-Library)
        subprocess.run(command, capture_output=True, text=True, check=True)
        return True

    except subprocess.CalledProcessError as e:
        # 2. The Fallback Path (Python Library)
        error_msg: str = e.stderr.strip()

        tqdm.write(
            f"\n[Warning] Strict parser failed on {midi_path.name}: {error_msg}"
        )
        tqdm.write("-> Routing to forgiving music21 fallback parser...")

        try:
            # Cast to Any to satisfy Pylance strict mode on untyped library boundaries
            score: Any = cast(Any, converter.parse(midi_path)) # type: ignore
            score.write("abc", fp=abc_path)
            tqdm.write("-> Fallback successful!")
            return True

        except Exception as fallback_err:
            tqdm.write(
                "-> Fallback also failed. "
                f"File is severely corrupted: {fallback_err}"
            )
            return False

    except FileNotFoundError:
        tqdm.write("\nCritical Error: 'midi2abc' command not found.")
        raise


def main() -> None:
    print(f"Scanning directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists before writing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files: list[Path] = list(INPUT_DIR.rglob("*.mid")) + list(
        INPUT_DIR.rglob("*.midi")
    )

    if not midi_files:
        print("No MIDI files found in the target directory.")
        return

    print(f"Found {len(midi_files)} MIDI files. Starting batch conversion...")

    success_count: int = 0

    for midi_file in tqdm(midi_files, desc="Converting MIDI to ABC"):
        success: bool = convert_midi_to_abc(midi_file)
        if success:
            success_count += 1

    print("\n" + "=" * 50)
    print("Batch Conversion Complete!")
    print(f"Successfully processed: {success_count} / {len(midi_files)} files.")
    print("=" * 50)


if __name__ == "__main__":
    main()