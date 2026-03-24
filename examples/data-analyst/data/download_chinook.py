#!/usr/bin/env python3
"""
Download the Chinook sample SQLite database.
Source: github.com/lerocha/chinook-database (MIT license)

Run once before using the data-analyst example:
    python data/download_chinook.py
"""
import pathlib
import sys
import urllib.request

URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
DEST = pathlib.Path(__file__).parent / "chinook.db"


def main() -> None:
    if DEST.exists():
        print(f"Already exists: {DEST}")
        return

    print("Downloading Chinook database...")
    urllib.request.urlretrieve(URL, DEST)
    size_kb = DEST.stat().st_size // 1024
    print(f"Saved to: {DEST} ({size_kb} KB)")
    print("Schema: albums, artists, customers, employees, genres,")
    print("        invoices, invoice_items, media_types, playlists,")
    print("        playlist_track, tracks")


if __name__ == "__main__":
    main()
