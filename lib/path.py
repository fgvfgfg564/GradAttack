import os
from pathlib import Path
import json

def make_folder(filename):
    pth = Path(filename)
    folder = pth.parent
    os.makedirs(folder, exist_ok=True)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def dump_json(obj, filename):
    make_folder(filename)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)