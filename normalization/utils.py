import numpy as np
import csv
import typing
import os


def normalize(words: np.ndarray, csv_path: str) -> None:
    norm = np.linalg.norm(words[:, 0])
    for i in words:
        i[0] = float(i[0]) / norm
    with open(csv_path, 'w') as new_file:
        csv_writer = csv.writer(new_file)

        for i in words:
            csv_writer.writerow(i)
