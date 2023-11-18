from datetime import timedelta
import re
import easyocr
import numpy as np
from PIL import Image
from collections import defaultdict
import os
import json

import pandas as pd

_reader = easyocr.Reader(['en', 'da'])

def readText(img: Image, width_ths: float=0.35) -> list[tuple[list[list[int]],str,float]]:
    allow_list = "abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ1234567890:. "
    return _reader.readtext(
        np.array(img), 
        detail=1,
        allowlist=allow_list, 
        decoder="greedy",
        beamWidth=8, 
        width_ths=width_ths) # Fine tuned. "Hoestrup" name is long and clashes with next column. This value causes the text boxes to be split

def _group_by_x_coordinate(matches: list[tuple[list[list[int]],str,float]]) -> defaultdict[int, list[tuple[list[list[int]], str]]]:
    columns = defaultdict(list)
    for box, text, _ in matches:
        top_left = box[0]
        x = top_left[0]
        columns[x].append((box, text))
    return columns

def _merge_keys(keys: list[int], threshold: int) -> list[list[int]]:
    current = []
    groups = []
    for k in sorted(keys):
        if len(current) == 0:
            current.append(k)
        else:
            prev = current[-1]
            diff = k - prev
            if diff < threshold:
                current.append(k)
            else:
                groups.append(current)
                current = [k]
    
    if len(current) > 0:
        groups.append(current)
    return groups
    
def _merge_columns(merged_keys: list[list[int]], column_dictionary: defaultdict[int, list[tuple[list[list[int]], str]]]) -> list[list[tuple[list[list[int]], str]]]:
    columns : list[list[tuple[list[list[int]], str]]] = []
    for group in merged_keys:
        current : list[tuple[list[list[int]], str]] = []
        for idx in group:
            current.extend(column_dictionary[idx])
        
        columns.append(sorted(current, key=lambda x: x[0][0][1])) # Sort by Y coordinate
    return columns
    
def _extract_columns(matches: list[tuple[list[list[int]],str,float]], threshold: int=30):
    columns = _group_by_x_coordinate(matches)
    merged_keys = _merge_keys(columns.keys(), threshold)
    return _merge_columns(merged_keys, columns)

def _apply_corrections(input_img_path: str, columns: list[tuple[list[list[int]],str,float]]) -> None:
    corrections_path = get_corrections_path(input_img_path)
    if os.path.exists(corrections_path):
        corrections = None
        with open(corrections_path, "r") as f:
            corrections = json.load(f)
        for column_idx, column_corr in corrections.items():
            if "delete" in column_corr:
                for row_to_delete in column_corr["delete"]:
                    _, value_to_delete = columns[int(column_idx)][int(row_to_delete)]
                    print(f"Removing row {row_to_delete} in column {column_idx} with value '{value_to_delete}'")
                    columns[int(column_idx)].pop(int(row_to_delete)) 
            for row_idx, new_value in column_corr.items():
                if row_idx == "delete":
                    continue
                box, old_value = columns[int(column_idx)][int(row_idx)]
                print(f"Applied correction in row {row_idx}, column {column_idx}, replace \t'{old_value}' with \t'{new_value}'")
                columns[int(column_idx)][int(row_idx)] = (box, new_value)

def _duration_parser(duration_str):
    parts = [p for p in re.split(r'[. :]', duration_str) if p]
    if len(parts) == 3:
        # The input string has the "minutes.seconds.milliseconds" format
        minutes, seconds, milliseconds = map(int, parts)
    elif len(parts) == 2:
        # The input string has the "seconds.milliseconds" format
        minutes = 0
        seconds, milliseconds = map(int, parts)
    else:
        raise ValueError("Invalid format " + duration_str)

    # Create a timedelta object
    #print(duration_str, minutes, seconds, milliseconds)
    return timedelta(
        minutes=minutes,
        seconds=seconds, 
        milliseconds=milliseconds)
    

def _get_kart_and_driver(column):
    box, text = column[0]
    pattern = r'(\d+): (\S+)'
    match = re.match(pattern, text)
    if match:
        return match.groups()

def _get_heat_as_table(heat, heat_name):
    rows = []
    for column in heat:
        kart, driver = _get_kart_and_driver(column)
        # Correct name
        if driver == 'Hoestru':
            driver = 'Hoestrup'
        for lap_num, (_, time) in enumerate(column[1:], 1):
            parsed_time = _duration_parser(time)
            rows.append((heat_name, driver, kart, lap_num, parsed_time))
    return rows,["Heat", "Driver", "Kart", "Lap", "Time"]

def get_corrections_path(input_img_path: str):
    base_dir, original_file = os.path.split(input_img_path)
    base_name, _ = os.path.splitext(original_file)
    corrections_filename = f"{base_name}-corrections.json"
    return os.path.join(base_dir, corrections_filename)


def read_scores(image_path: str, width_ths: float=0.35, column_threshold: int=30):
    with Image.open(image_path) as image:
        boxes = readText(image, width_ths=width_ths)
        columns = _extract_columns(boxes, column_threshold)
        _apply_corrections(image_path, columns)
        return columns
    
def read_to_pandas(image_path: str, heat_name: str):
    columns = read_scores(image_path)
    table, header = _get_heat_as_table(columns, heat_name)
    return pd.DataFrame(table, columns=header)

def display_row_wise(columns,row=None):
    end = max(len(c) for c in columns)
    for r in range(0, end):
        values = [c[r][1] if len(c) > r else None for c in columns]
        if row is None or row == r:
            print(values)