import json
import os

# ensure_dir function taken from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory) and directory.strip():
        os.makedirs(directory)

# jsonKeys2Int taken from stackoverfolw answer https://stackoverflow.com/a/34346202
def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def readJsonFromFile(filename):
    with open(filename) as fp:
        situationFileJson = json.load(fp)
    return situationFileJson 

def writeJsonToFile(filename, jsonToWrite, indent = 4):
    ensure_dir(filename)
    with open(filename, 'w') as outfile:
        json.dump(jsonToWrite, outfile, indent = indent)
        outfile.write("\n")
        
def jsonToString(jsonToWrite, indent = 4):
    return json.dumps(jsonToWrite, indent = indent)