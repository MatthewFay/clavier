import argparse
import concurrent.futures
import subprocess
from pathlib import Path

# This script will take a directory of raw ABC files and
# create 12 versions of each file, transposed by 0 to 11 half-steps.
# This helps prevent overfitting and increases the diversity of the training data
# for our music generation model.


def transpose_file(abc_path: Path, output_dir: Path) -> list[str]:
    """Transposes an ABC file natively using the abc2abc CLI tool."""
    generated_files: list[str] = []

    for half_steps in range(1, 12):
        new_filename = f"{abc_path.stem}_+{half_steps}.abc"
        output_path = output_dir / new_filename

        try:
            # -e: Ignore strict syntax warnings, -t: transpose by X semitones
            result = subprocess.run(
                ["abc2abc", str(abc_path), "-e", "-t", str(half_steps)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Write the transposed ABC text to the new file
            output_path.write_text(result.stdout, encoding="utf-8")
            generated_files.append(new_filename)

        except subprocess.CalledProcessError:
            # Skip this transposition if the C-tool completely fails to parse
            pass

    # Copy the original file (+0)
    original_output = output_dir / f"{abc_path.stem}_+0.abc"
    original_output.write_bytes(abc_path.read_bytes())
    generated_files.append(original_output.name)

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment ABC datasets via abc2abc transposition."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to raw ABC files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save augmented ABC files"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    abc_files = list(input_dir.rglob("*.abc"))
    print(f"Found {len(abc_files)} files. Transposing using abcmidi C-library...")

    successful_files = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(transpose_file, abc, output_dir): abc for abc in abc_files
        }

        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            successful_files += len(results)
            print(
                f"\rGenerated {successful_files} augmented files...", end="", flush=True
            )

    print(f"\n\nAugmentation Complete: {output_dir}")


if __name__ == "__main__":
    main()
