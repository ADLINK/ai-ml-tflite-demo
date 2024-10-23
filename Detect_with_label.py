import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on.', default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on.', default=None)
parser.add_argument('--noshow_results', help='Don\'t show result images (useful for headless systems)', action='store_true')
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
save_results = args.save_results
show_results = not args.noshow_results
IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if IM_NAME and IM_DIR:
    print('Error! Please use either --image or --imagedir.')
    sys.exit()

if not IM_NAME and not IM_DIR:
    IM_NAME = 'test1.jpg'

# Import TensorFlow Lite libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to the current working directory
CWD_PATH = os.getcwd()

# Define path to images
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
else:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_NAME)
    images = [PATH_TO_IMAGES]

# Load the label map
with open(os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME), 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove '???' label if using COCO model
if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Determine the output indexes for the model
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Process each image
for image_path in images:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Debugging: print detected object and confidence
            object_name = labels[int(classes[i])]  # Get the object name
            print(f"Detected {object_name} with confidence {scores[i]*100:.2f}%")

            # Draw label on the image
            label = f"{object_name}: {int(scores[i]*100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if show_results:
        cv2.imshow('Detection Results', image)
        if cv2.waitKey(0) == ord('q'):
            break

    if save_results:
        result_path = os.path.join(CWD_PATH, 'results', os.path.basename(image_path))
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        cv2.imwrite(result_path, image)

cv2.destroyAllWindows()
