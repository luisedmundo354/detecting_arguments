import os, sys, pathlib

prefix = "Zaid_"
folder = pathlib.Path("label_studio_taks/zaidal-huneidi@umaryland.edu")

for path in folder.iterdir():
    if path.is_file():
        new_name = path.with_name(prefix + path.name)
        path.rename(new_name)