import sys
import os
import re

CURR_DIR = sys.argv[1]

frames = [name for name in os.listdir(CURR_DIR) if not re.match(r'__', name)]

print(len(frames))
