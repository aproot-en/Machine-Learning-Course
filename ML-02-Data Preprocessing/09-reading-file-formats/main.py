


import os
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from load_data import load_data, load_pixels, OUTPUT_DIR
from plot_result import plot_file_sizes, plot_image

FILES_DIR = os.path.join(OUTPUT_DIR, "files")


def main():
    print("-" * 30)
    print("09 - Reading CSV, JSON, TXT, Image, and SQLite Files")
    print("-" * 30)

    os.makedirs(FILES_DIR, exist_ok=True)
    df = load_data().head(200)
    sizes = {}

    print("\n[1] CSV - Text Table is the easiest to read")
    path = os.path.join(FILES_DIR, "sample.csv")
    df.to_csv(path, index=False)

    from_csv = __import__("pandas").read_csv(path)
    sizes["CSV"] = os.path.getsize(path)
    print(f"Written and read back: {len(from_csv)} rows, {len(from_csv.columns)} columns")

    print("\n[2] JSON - Stores nested structures, larger files due to key names")
    path = os.path.join(FILES_DIR, "sample.json")
    df.to_json(path, orient="records", indent=2)

    from_json = __import__("pandas").read_json(path)
    sizes["JSON"] = os.path.getsize(path)
    print(f"Written and read back: {len(from_json)} rows, {len(from_json.columns)} columns")

    print("\n[3] TXT - Pure text, need to specify delimiter")
    path = os.path.join(FILES_DIR, "sample.txt")
    df.to_csv(path, sep="\t", index=False)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    sizes["TXT"] = os.path.getsize(path)
    print(f"Written and read back: {len(lines)} lines")
    print(f"First line: {lines[0].strip()}")

    print("\n[4] Image - ภาพ อ่านมาเป็นเมทริกซ์ตัวเลข")
    images, _ = load_pixels(5)
    path = os.path.join(FILES_DIR, "face.png")
    plt.imsave(path, images[0], cmap="gray")

    from_image = mpimg.imread(path)
    sizes["PNG"] = os.path.getsize(path)
    print(f"Written and read back: shape {from_image.shape}")

    print("\n[5] SQLite - dataset in a single file database, can be queried with SQL")
    path = os.path.join(FILES_DIR, "sample.db")
    if os.path.exists(path):
        os.remove(path)

    connection = sqlite3.connect(path)
    df.to_sql("people", connection, index=False)

    pd = __import__("pandas")
    query = pd.read_sql("SELECT gender, COUNT(*) AS n FROM people GROUP BY gender",
                        connection)
    connection.close()

    sizes["SQLite"] = os.path.getsize(path)
    print("result from SQL:")
    print(query.to_string(index=False))

    print("\n[6] which format to choose")
    print(f"{'Format':<10}{'Size (KB)':>12}   {'Best For'}")
    notes = {
        "CSV": "General tabular data, can be opened with Excel",
        "JSON": "Nested structures, suitable for API communication",
        "TXT": "Raw text data or logs",
        "PNG": "Image data",
        "SQLite": "Large datasets requiring querying",
    }
    for name, size in sizes.items():
        print(f"{name:<10}{size / 1024:>12.1f}   {notes[name]}")

    print("\n[7] plot graphs")
    plot_file_sizes(sizes, OUTPUT_DIR)
    plot_image(from_image, OUTPUT_DIR)


if __name__ == "__main__":
    main()
