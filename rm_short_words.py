import json
import sys
import os
import re

CURR_DIR = sys.argv[1]
# print(CURR_DIR)

subs_path = [name for name in os.listdir(CURR_DIR) if re.match(r'__', name)][0]
with open(os.path.join(CURR_DIR, subs_path), 'r') as subs_file:
    subs = str(json.loads(subs_file.read())['word']).lower()
    print(len(subs))
