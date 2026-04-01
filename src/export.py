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
    midi_cmd = ["abc2midi", str(abc_path), "-o", str(mid_path)]
    subprocess.run(
        midi_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    # 2. Convert ABC to PostScript using abcm2ps
    print(" -> Generating Sheet Music (PostScript)...")
    ps_cmd = ["abcm2ps", str(abc_path), "-O", str(ps_path)]
    subprocess.run(
        ps_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    # 3. Convert PostScript to PDF using ps2pdf (Ghostscript)
    print(" -> Compiling to PDF...")
    pdf_cmd = ["ps2pdf", str(ps_path), str(pdf_path)]
    subprocess.run(pdf_cmd, check=True)

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
        print("Ensure you have installed the required system tools:")
        print("sudo apt-get install abcmidi abcm2ps ghostscript")


if __name__ == "__main__":
    main()
