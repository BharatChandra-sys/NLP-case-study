import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "bharatchandra1234"
DATASET = f"{USER}/telelens-code"
COMCAST = "amirizzudin4/comcast-consumer-complaints"
KERNELS = {
    "01_word2vec": "telelens-01-word2vec",
    "02_sentiment_classification": "telelens-02-sentiment",
    "03_text_summarization": "telelens-03-summarization",
    "04_factoid_question_answering": "telelens-04-factoid-qa",
    "05_integrated_system": "telelens-05-integrated",
}


def kaggle(*args):
    return subprocess.run([sys.executable, "-m", "kaggle", *args], text=True, encoding="utf-8", errors="replace")


def push_code(message):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests", tmp / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        (tmp / "data" / "gold").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "gold" / "qa_gold.jsonl", tmp / "data" / "gold" / "qa_gold.jsonl")
        meta = {"title": "telelens-code", "id": DATASET, "licenses": [{"name": "MIT"}]}
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta))
        status = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", DATASET], capture_output=True, text=True)
        if status.returncode == 0 and "ready" in status.stdout.lower():
            kaggle("datasets", "version", "-p", str(tmp), "-m", message, "-r", "zip", "-q")
        else:
            kaggle("datasets", "create", "-p", str(tmp), "-r", "zip", "-q")


def push_kernel(name):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(ROOT / "notebooks" / f"{name}.ipynb", tmp / f"{name}.ipynb")
        meta = {
            "id": f"{USER}/{KERNELS[name]}", "title": KERNELS[name], "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [DATASET, COMCAST], "competition_sources": [], "kernel_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta))
        kaggle("kernels", "push", "-p", str(tmp))


def status(names):
    for name in names:
        kaggle("kernels", "status", f"{USER}/{KERNELS[name]}")


def pull(name):
    target = ROOT / "kaggle_outputs" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    kaggle("kernels", "output", f"{USER}/{KERNELS[name]}", "-p", str(target), "-o")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["code", "push", "status", "pull"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("-m", default="update")
    args = parser.parse_args()
    names = [n for n in KERNELS if not args.names or any(n.startswith(a) for a in args.names)]
    if args.action == "code":
        push_code(args.m)
    elif args.action == "push":
        for name in names:
            push_kernel(name)
    elif args.action == "status":
        status(names)
    else:
        for name in names:
            pull(name)


if __name__ == "__main__":
    main()
