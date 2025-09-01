# logs/logger.py
import csv, os

class CSVLogger:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header); self.f.flush()
    def row(self, *vals):
        self.w.writerow(vals); self.f.flush()
    def close(self): self.f.close()
