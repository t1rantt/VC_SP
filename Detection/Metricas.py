import cv2
import argparse

# Function to calculate the percentage of similarity between two bounding boxes
def calcular_similitud(bbox1, bbox2):
    # Calculate the intersection area between the bounding boxes
    inter_area = max(0, min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]) - max(bbox1[0], bbox2[0])) * \
                 max(0, min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]) - max(bbox1[1], bbox2[1]))
    
    # Calculate the union area of the bounding boxes
    union_area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - inter_area
    
    # Calculate the percentage of similarity
    similitud = inter_area / union_area
    
    return similitud


# Function to read the data of the bounding boxes from a file
def leer_datos_from_archivo(archivo):
    datos = []
    with open(archivo, 'r') as f:
        for linea in f:
            datos.append([int(x) for x in linea.strip().split(',')])
    return datos


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("archivo_correctas", help="Path to the file with correct bounding boxes.")
parser.add_argument("archivo_tracker", help="Path to the file with tracker bounding boxes.")
args = parser.parse_args()

# Read the data from the files
datos_correctas = leer_datos_from_archivo(args.archivo_correctas)
datos_tracker = leer_datos_from_archivo(args.archivo_tracker)

# Variables for the counter and the first frame with similarity less than 50%
contador_frames = 0
primer_frame_menor_50 = None

# Compare bounding boxes line by line
for i in range(len(datos_correctas) - 1):
    bbox_correcta = datos_correctas[i][1:5]
    bbox_tracker = datos_tracker[i][1:5]
    
    similitud = calcular_similitud(bbox_correcta, bbox_tracker)
    
    if similitud >= 0.5:
        contador_frames += 1
    
    if similitud < 0.25 and primer_frame_menor_50 is None:
        primer_frame_menor_50 = datos_correctas[i][0]

# Print results
print(contador_frames)
print(primer_frame_menor_50)
