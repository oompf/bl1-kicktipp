#!/usr/bin/env python3

import urllib.request, json, glob
from match import Match, MatchList
import sys

if __name__ == "__main__":
    match_list = MatchList()

    # 1. Alte Spieldaten laden
    for fn in glob.glob("data/*.json"):
        with open(fn) as f:
            j = json.loads(f.read())
            for el in j:
                match_list.append(Match.from_json(el))

    # 2. Daten für die aktuelle Saison laden
    with urllib.request.urlopen("https://www.openligadb.de/api/getmatchdata/bl1/2019/") as f:
        j = json.loads(f.read().decode('utf-8'))
        for el in j:
            match_list.append(Match.from_json(el))
    
    # 3. Daten verarbeiten
    match_list.compute()
    match_list.predict_upcoming()

    if "--table" in sys.argv:
        print(match_list.elo_table)