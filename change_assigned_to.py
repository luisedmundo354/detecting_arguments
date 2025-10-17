import json, pathlib, shutil, argparse

new = "zaidal-huneidi@umaryland.edu"

def main(path=".", glob="*.json", value=new):
    folder = pathlib.Path(path)
    print(folder, folder.exists())
    for file in folder.glob(glob):
        if not file.is_file():
            print(f"Skipping {file}")
            continue
        print(f"Processing {file}")
        data = json.loads(file.read_text(encoding="utf-8"))
        data.setdefault("data", {})["assigned_to"] = value

        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        shutil.move(tmp, file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", nargs="?")
    p.add_argument("--glob", default="*.json")
    p.add_argument("--value", default=new)
    main(**vars(p.parse_args()))