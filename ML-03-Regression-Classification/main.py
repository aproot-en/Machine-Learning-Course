import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = ["regression", "classification"]


def main():
    # Separate processes: both folders have main.py and model.py with the
    # same names, so importing them together would clash
    for task in TASKS:
        print("\n" + "#" * 60)
        print(f"# {task}")
        print("#" * 60 + "\n")

        subprocess.run([sys.executable,
                        os.path.join(BASE_DIR, task, "main.py")])


if __name__ == "__main__":
    main()
