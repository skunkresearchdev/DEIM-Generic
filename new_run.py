import argparse
import re  # Import re for regular expressions
import subprocess
import sys
from pathlib import Path

# Define ANSI color codes
COLOR_BLUE = "\033[94m"  # Bright Blue
COLOR_RED = "\033[91m"  # Bright Red
COLOR_GREEN = "\033[92m"  # Bright Green
COLOR_WHITE = "\033[97m"  # Bright White
COLOR_END = "\033[0m"  # Reset color


def find_latest_run_dir(base_dir_str):
    """
    Finds the latest timestamped subdirectory (YYYYMMDD_HHMMSS) in the base directory.
    Uses pathlib.iterdir() and regex filtering as a robust alternative to glob.

    Args:
        base_dir_str (str): The base directory path (e.g., 'deim_outputs/under').

    Returns:
        str: The path to the latest run directory, or None if not found.
    """
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        print(
            f"{COLOR_RED}Error: Base directory not found: {base_dir}{COLOR_END}",
            file=sys.stderr,
        )
        return None
    if not base_dir.is_dir():
        print(
            f"{COLOR_RED}Error: Base path is not a directory: {base_dir}{COLOR_END}",
            file=sys.stderr,
        )
        return None

    # Timestamp pattern regex: Matches YYYYMMDD_HHMMSS
    # \d{2} is equivalent to [0-9]{2}
    timestamp_pattern_regex = re.compile(r"20\d{2}[0-1]\d[0-3]\d_[0-2]\d[0-5]\d[0-5]\d")

    run_dirs = []
    try:
        # Iterate over entries in the base directory
        for entry in base_dir.iterdir():
            # Check if the entry is a directory and its name matches the timestamp pattern
            if entry.is_dir() and timestamp_pattern_regex.fullmatch(entry.name):
                run_dirs.append(entry)

    except Exception as e:
        # Catch potential errors during directory iteration
        print(
            f"{COLOR_RED}Error during directory iteration or regex matching: {e}{COLOR_END}",
            file=sys.stderr,
        )
        return None

    if not run_dirs:
        print(
            f"{COLOR_RED}Error: No timestamped directories matching the pattern found in {base_dir}{COLOR_END}",
            file=sys.stderr,
        )
        return None

    # Sort the found directories by path (which includes the timestamp)
    # The latest directory will be the last one after sorting
    latest_dir = sorted(run_dirs)[-1]
    print(f"Found latest run directory: {latest_dir}")
    # Return as string path for subprocess compatibility
    return str(latest_dir)


def find_weight_file(run_dir_str):
    """
    Finds the weight file in the specified run directory, prioritizing converted files.
    Search order: best_stg2_converted.pth, best_stg1_converted.pth,
                  best_stg2.pth, best_st1.pth.

    Args:
        run_dir_str (str): The path to the run directory.

    Returns:
        str: The path to the found weight file, or None if none of the expected files are found.
    """
    run_dir = Path(run_dir_str)
    if not run_dir.is_dir():
        print(
            f"{COLOR_RED}Error: Run directory not found or is not a directory: {run_dir}{COLOR_END}",
            file=sys.stderr,
        )
        return None

    # Define potential weight file paths in order of preference
    potential_files = [
        run_dir / "best_stg2_converted.pth",
        run_dir / "best_stg1_converted.pth",
        run_dir / "best_stg2.pth",
        run_dir / "best_st1.pth",
    ]

    # Check for existence in the defined order
    for file_path in potential_files:
        if file_path.exists():
            print(f"Found weight file: {file_path}")
            return str(file_path)  # Return as string path

    # If none of the files were found
    print(
        f"{COLOR_RED}Error: None of the expected weight files found in {run_dir}.{COLOR_END}",
        file=sys.stderr,
    )
    print(f"Looked for: {[str(f.name) for f in potential_files]}", file=sys.stderr)
    return None


def run_command(command, cwd=None):
    """
    Runs a shell command using subprocess and captures output.
    Prints errors in red on failure. Does NOT print stdout/stderr on success.

    Args:
        command (list): A list of strings representing the command and its arguments.
        cwd (str, optional): The current working directory for the command. Defaults to None.

    Returns:
        tuple: (bool success, str stdout, str stderr).
               success is True if the command finished with exit code 0, False otherwise.
    """
    # Print the command being executed for clarity
    print(f"\nRunning command: {' '.join(command)}")
    try:
        # Use run with check=True to raise CalledProcessError on non-zero exit code
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout/stderr as text
        result = subprocess.run(
            command, check=True, cwd=cwd, capture_output=True, text=True
        )
        # On success, return the captured output
        return (True, result.stdout, result.stderr)

    except FileNotFoundError:
        error_msg = f"{COLOR_RED}Error: Command not found. Make sure the executable is in your PATH: {command[0]}{COLOR_END}"
        print(error_msg, file=sys.stderr)
        return (False, "", error_msg)
    except subprocess.CalledProcessError as e:
        print(
            f"{COLOR_RED}Error: Command failed with exit code {e.returncode}{COLOR_END}",
            file=sys.stderr,
        )
        print(f"{COLOR_RED}Command output:{COLOR_END}\n{e.stdout}", file=sys.stderr)
        print(
            f"{COLOR_RED}Command error output:{COLOR_END}\n{e.stderr}", file=sys.stderr
        )
        return (False, e.stdout, e.stderr)
    except Exception as e:
        error_msg = f"{COLOR_RED}An unexpected error occurred while running command: {e}{COLOR_END}"
        print(error_msg, file=sys.stderr)
        return (False, "", error_msg)


def main():
    """Main function to parse arguments and execute the workflow steps."""
    parser = argparse.ArgumentParser(
        description="Automate training, weight conversion, and ONNX export workflow."
    )
    parser.add_argument(
        "option",
        choices=["under", "sides"],
        help="Specify 'under' or 'sides' training configuration. Determines config file and output directory.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Specify the starting step (1: Train, 2: Convert, 3: Export). Defaults to 1.",
    )
    args = parser.parse_args()

    # Determine configuration file and output base directory based on the option
    config_file = f"deim_model_{args.option}.yml"
    output_base_dir = f"deim_outputs/{args.option}"

    # --- Determine which steps to run based on --start-step ---
    run_step_1 = args.start_step <= 1
    run_step_2 = args.start_step <= 2
    run_step_3 = args.start_step <= 3

    latest_run_dir = None
    weight_file_path = None

    # --- Step 1: Run Training (Conditional) ---
    if run_step_1:
        print(f"\n{COLOR_BLUE}--- Starting Step 1: Training ---{COLOR_END}")
        train_command = [
            "torchrun",
            "--master_port=7777",
            "--nproc_per_node=1",
            "train.py",
            "-c",
            config_file,
            "--use-amp",
            "--seed=0",
            "-t",
            "./base_model.pth",
        ]
        success, stdout, stderr = run_command(train_command)

        if success:
            print("Command output:\n", stdout)
            if stderr:
                # Print stderr normally for informational purposes if not empty
                print("Command error output:\n", stderr, file=sys.stderr)
            print("Command finished successfully.")
            print(f"{COLOR_BLUE}--- Step 1: Training Finished ---{COLOR_END}")
        else:
             # Error details printed by run_command
            print(
                f"{COLOR_RED}Step 1 (Training) failed. Aborting workflow.{COLOR_END}",
                file=sys.stderr,
            )
            sys.exit(1)


    # --- Find the latest run directory and weight file ---
    # This needs to run before step 2 or 3, regardless of the start step,
    # as these steps depend on the paths found.
    print(
        f"\n{COLOR_BLUE}--- Preparing for Conversion/Export: Finding latest run directory and weight file ---{COLOR_END}"
    )
    latest_run_dir = find_latest_run_dir(output_base_dir)
    if not latest_run_dir:
        print(
            f"{COLOR_RED}Preparation failed: Could not find the latest run directory. Aborting workflow.{COLOR_END}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find the weight file, prioritizing converted ones
    weight_file_path = find_weight_file(latest_run_dir)
    if not weight_file_path:
        print(
            f"{COLOR_RED}Preparation failed: Could not find a suitable weight file in the latest run directory. Aborting workflow.{COLOR_END}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"{COLOR_BLUE}--- Preparation Complete ---{COLOR_END}")

    # --- Step 2: Convert Weight (Conditional) ---
    # Note: convert_weight.py seems to take the directory as input (--checkpoint_dir)
    # and likely produces a _converted.pth file within that directory.
    # The export_onnx.py step (Step 3) will now use the converted file if found by find_weight_file.
    if run_step_2:
        print(f"\n{COLOR_BLUE}--- Starting Step 2: Weight Conversion ---{COLOR_END}")
        convert_command = [
            "python",
            "tools/reference/convert_weight.py",
            "--checkpoint_dir",
            latest_run_dir,  # Use the directory found
            "--filter",
            "stg2",  # Keep the filter as in the original command
        ]
        success, stdout, stderr = run_command(convert_command)

        if success:
            print("Command output:\n", stdout)
            if stderr:
                 # Print stderr normally for informational purposes if not empty
                print("Command error output:\n", stderr, file=sys.stderr)
            print("Command finished successfully.")
            print(f"{COLOR_BLUE}--- Step 2: Weight Conversion Finished ---{COLOR_END}")
        else:
             # Error details printed by run_command
            print(
                f"{COLOR_RED}Step 2 (Weight Conversion) failed. Aborting workflow.{COLOR_END}",
                file=sys.stderr,
            )
            sys.exit(1)


    # --- Step 3: Export ONNX (Conditional) ---
    # Note: export_onnx.py uses the config file (-c) and the specific weight file (-r).
    # find_weight_file now ensures weight_file_path points to the preferred file (converted or original).
    if run_step_3:
        print(f"\n{COLOR_BLUE}--- Starting Step 3: ONNX Export ---{COLOR_END}")
        export_command = [
            "python",
            "tools/deployment/export_onnx.py",
            "-c",
            config_file,
            "-r",
            weight_file_path,  # Use the specific file path found (converted or original)
        ]
        # Run the export command
        success, stdout, stderr = run_command(export_command)

        if success:
            print("Command output:\n", stdout)
            # Suppress stderr output for Step 3 on success
            print("Command finished successfully.")
            print(f"{COLOR_BLUE}--- Step 3: ONNX Export Finished ---{COLOR_END}")

            # --- Rename/Move ONNX Output File ---
            # Based on user feedback, the ONNX file is created in the run directory
            # and named based on the input weight file name (e.g., best_stg2_converted.onnx).
            # We need to move it to the current directory and rename it to <option>.onnx.

            # Construct the expected path to the ONNX file in the run directory
            # This replaces the .pth extension of the weight_file_path with .onnx
            expected_onnx_in_run_dir = Path(weight_file_path).with_suffix(".onnx")

            # Define the target path in the current directory
            target_output_name = f"{args.option}.onnx"
            target_output_path = Path(target_output_name)  # Target in current directory

            if expected_onnx_in_run_dir.exists():
                try:
                    # Move and rename the file from the run directory to the target path
                    expected_onnx_in_run_dir.rename(target_output_path)
                    # Formatted output for successful rename/move
                    print(
                        f"{COLOR_WHITE}Moved and renamed exported ONNX file to: {COLOR_GREEN}{target_output_path}{COLOR_END}"
                    )
                except OSError as e:
                    print(
                        f"{COLOR_RED}Error moving/renaming ONNX file {expected_onnx_in_run_dir} to {target_output_path}: {e}{COLOR_END}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"{COLOR_RED}An unexpected error occurred during ONNX file renaming: {e}{COLOR_END}",
                        file=sys.stderr,
                    )
            else:
                # Error if the expected ONNX file is not found in the run directory
                print(
                    f"{COLOR_RED}Error: Expected ONNX output file '{expected_onnx_in_run_dir}' not found after export in the run directory. Cannot move/rename.{COLOR_END}",
                    file=sys.stderr,
                )
        else:
             # Error details printed by run_command
            print(
                f"{COLOR_RED}Step 3 (ONNX Export) failed. Aborting workflow.{COLOR_END}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\nWorkflow execution complete.")
    # Report which steps were attempted/executed
    steps_attempted = []
    if args.start_step <= 1:
        steps_attempted.append("Step 1 (Train)")
    if args.start_step <= 2:
        steps_attempted.append("Step 2 (Convert)")
    if args.start_step <= 3:
        steps_attempted.append("Step 3 (Export)")

    if not steps_attempted:
        print("No steps were selected to run based on --start-step.")
    else:
        print(
            f"Started from Step {args.start_step}. Attempted steps: {', '.join(steps_attempted)}"
        )


if __name__ == "__main__":
    main()