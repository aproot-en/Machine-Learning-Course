


import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOPICS = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d[0].isdigit()
)

def main():
    for topic in TOPICS:
        print("\n" + "--" * 30)
        print(f"-- {topic}")
        print("--" * 30 + "\n")

        subprocess.run([sys.executable, "main.py"],
                       cwd=os.path.join(BASE_DIR, topic))


if __name__ == "__main__":
    main()
