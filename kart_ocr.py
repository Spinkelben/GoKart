import easyocr
import numpy as np
from PIL import Image
from collections import defaultdict

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

def read_scores(image: Image, width_ths: float=0.35, column_threshold: int=30):
    boxes = readText(image, width_ths=width_ths)
    return _extract_columns(boxes, column_threshold)