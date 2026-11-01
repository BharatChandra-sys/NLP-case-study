import glob
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

WE_BASE = "https://raw.githubusercontent.com/HeshamAsem/WEDataset/main/Binary Mixed Dataset/"
FEWSUM_BASE = "https://raw.githubusercontent.com/abrazinskas/FewSum/master/artifacts/amazon/gold_summs/"
SQUAD_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SIMLEX = "https://fh295.github.io/SimLex-999.zip"
COMCAST_FILE = "comcast_consumeraffairs_complaints.csv"
COMCAST_REF = "amirizzudin4/comcast-consumer-complaints"


def data_dir():
    default = "/tmp/telelens-data" if os.path.isdir("/kaggle/working") else str(Path(__file__).resolve().parents[2] / "data" / "external")
    path = Path(os.environ.get("TELELENS_DATA", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url, name):
    target = data_dir() / name
    if not target.exists():
        safe = urllib.parse.quote(url, safe=":/")
        with urllib.request.urlopen(safe, timeout=120) as response:
            target.write_bytes(response.read())
    return target


def load_telecom(size=10000):
    name = {1000: "Data1KMixed.csv", 5000: "Data5KMixed.csv", 10000: "Data10KMixed.csv"}[size]
    frame = pd.read_csv(fetch(WE_BASE + name, name)).dropna(subset=["CleanedText", "Type"])
    frame = frame[frame["Type"].isin(["Bad", "Good"])]
    conflicts = frame.groupby("CleanedText")["Type"].nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    frame = frame[~frame["CleanedText"].isin(conflicts)].reset_index(drop=True)
    return pd.DataFrame({
        "text": frame["CleanedText"].astype(str),
        "label": (frame["Type"] == "Good").astype(int),
        "rating": frame["Rating"],
    })


def load_comcast():
    hits = glob.glob(f"/kaggle/input/**/{COMCAST_FILE}", recursive=True)
    local = data_dir() / "comcast" / COMCAST_FILE
    if hits:
        path = hits[0]
    elif local.exists():
        path = local
    else:
        raise FileNotFoundError(f"Attach Kaggle dataset {COMCAST_REF} or download it to {local}")
    frame = pd.read_csv(path)
    frame = frame[frame["text"].notna()]
    frame = frame.assign(doc_id=frame.index, text=frame["text"].astype(str))
    return frame.reset_index(drop=True)


def load_fewsum(split):
    frame = pd.read_csv(fetch(f"{FEWSUM_BASE}{split}.csv", f"fewsum_{split}.csv"), sep="\t")
    records = []
    for _, row in frame.iterrows():
        records.append({
            "group": row["group_id"],
            "category": row["cat"],
            "reviews": [str(row[f"rev{i}"]) for i in range(1, 9)],
            "ratings": [float(row[f"rating{i}"]) for i in range(1, 9)],
            "references": [str(row[f"summ{i}"]) for i in range(1, 4)],
        })
    return records


def load_squad():
    payload = json.loads(fetch(SQUAD_DEV, "dev-v1.1.json").read_text(encoding="utf-8"))
    rows = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": paragraph["context"],
                    "answers": [a["text"] for a in qa["answers"]],
                })
    return rows


def load_simlex():
    archive = zipfile.ZipFile(io.BytesIO(fetch(SIMLEX, "SimLex-999.zip").read_bytes()))
    member = next(n for n in archive.namelist() if n.endswith("SimLex-999.txt"))
    frame = pd.read_csv(archive.open(member), sep="\t")
    return frame[["word1", "word2", "SimLex999"]]


def load_gold(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
