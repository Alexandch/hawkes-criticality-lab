"""Fetch the two legacy archives required by a fresh repository clone."""
from pathlib import Path
from download_expanded_data import OSF,download_code,get_json


def main():
    target=Path('data/raw');target.mkdir(parents=True,exist_ok=True)
    listing=get_json(OSF)
    entries={entry['attributes']['name']:entry for entry in listing['data']}
    for year in (2020,2021):
        name=f'crime_open_database_core_{year}.csv.gz'
        if (target/name).exists():
            print(f'Using existing archive: {target/name}')
        else:
            download_code(entries[name],target)


if __name__=='__main__':main()
