import os
import datetime
# DEST_DIR = "~/workspace/scans"

import os
import time
import subprocess

SCAN_DIR = "~/Documents/scans/inbox"

# I want to do somethin where I check the photos I took today, if they "look like" a notebook photo, I move them into that folder. 

# I should probably do this with cron.
while True:
    for f in os.listdir(os.path.expanduser(SCAN_DIR)):
        if f.endswith(".pdf"):
            path = os.path.join(os.path.expanduser(SCAN_DIR), f)

            subprocess.run(["ocrmypdf", path, path])
            print("Processed:", f)

    time.sleep(60)

# THis is for renaming
for file in os.listdir(SCAN_DIR):
    if file.endswith(".pdf"):
        date = datetime.date.today().isoformat()
        new_name = f"{date}_notes.pdf"
        os.rename(
            os.path.join(SCAN_DIR, file),
            os.path.join(DEST_DIR, new_name)
        )