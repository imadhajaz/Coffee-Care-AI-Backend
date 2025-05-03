
# Function to show Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import re
import os
import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image
from text_gneration import get_image

def process_image_and_extract_text(var):
    output = subprocess.run(['yolo', 'segment', 'predict', f'model=./runs/best.pt', f'source="./infrence/{var}"', 'save=True'], capture_output=True, text=True)

    # Extract the predict number from the output
    matches = re.search(r'predict(\d+)', output.stdout)
    if matches:
        predict_number = int(matches.group(1)) 
        result_path = f"runs/segment/predict{predict_number}"
    else:
        print("Predict number not found in output.")
        result_path = None

    filename = os.path.basename(var)

    if result_path and filename:
        segmented_image_path = os.path.join(result_path, filename)
        print("Segmented image path:", segmented_image_path)
    else:
        print("Unable to construct segmented image path.")

    reader = easyocr.Reader(['en'])
    output = reader.readtext(segmented_image_path)
    result = ["".join(filter(str.isalpha, item[1])) for item in output if item[1].split()[0] in {'Rust', 'Miner', 'Phoma'}]

    return result,output