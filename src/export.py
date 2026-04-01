import argparse
import subprocess
from pathlib import Path


def abc_to_audio_and_sheet(abc_file_path: str | Path) -> None:
    """Converts an ABC file to a MIDI audio file and a PDF sheet music file."""
    abc_path = Path(abc_file_path).resolve()

    if not abc_path.exists():
        raise FileNotFoundError(f"Cannot find ABC file: {abc_path}")

    base_name = abc_path.stem
    output_dir = abc_path.parent

    mid_path = output_dir / f"{base_name}.mid"
    ps_path = output_dir / f"{base_name}.ps"
    pdf_path = output_dir / f"{base_name}.pdf"

    print(f"Processing: {abc_path.name}")

    # 1. Convert ABC to MIDI using abc2midi
    print(" -> Generating MIDI audio...")
    try:
        subprocess.run(
            ["abc2midi", str(abc_path), "-o", str(mid_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Warning: abc2midi had issues:\n{e.stderr}")

    # 2. Convert ABC to PostScript using abcm2ps
    print(" -> Generating Sheet Music (PostScript)...")
    try:
        subprocess.run(
            ["abcm2ps", str(abc_path), "-O", str(ps_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("\n❌ abcm2ps failed to compile the sheet music. Error log:")
        print("--------------------------------------------------")
        print(e.stderr)
        print("--------------------------------------------------")
        print("This usually means the AI generated slightly invalid ABC syntax.")
        print("Open the .abc file and check the line number mentioned above!")
        return

    # 3. Convert PostScript to PDF using ps2pdf (Ghostscript)
    print(" -> Compiling to PDF...")
    try:
        subprocess.run(
            ["ps2pdf", str(ps_path), str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ps2pdf failed:\n{e.stderr}")
        return

    # Cleanup the temporary PostScript file
    if ps_path.exists():
        ps_path.unlink()

    print("\n✅ Export Complete!")
    print(f"Audio: {mid_path}")
    print(f"Sheet: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ABC files to MIDI and PDF")
    parser.add_argument("file", type=str, help="Path to the .abc file")
    args = parser.parse_args()

    try:
        abc_to_audio_and_sheet(args.file)
    except Exception as e:
        print(f"Error during export: {e}")


if __name__ == "__main__":
    main()
