import json
from typing import Dict

def read(json_path: str) -> Dict:
    with open(json_path, 'r') as file:
        return json.load(file)

def write(json_path: str, content: Dict) -> Dict:
    with open(json_path, 'w') as file:
        json.dump(content, file)