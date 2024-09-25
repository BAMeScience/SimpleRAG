from PIL import Image, ImageChops, ImageDraw
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection

import torch
import matplotlib.pyplot as plt
import os
import psutil
import time
from transformers import DetrFeatureExtractor

import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import cv2

feature_extractor = DetrFeatureExtractor()
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
ocr = PaddleOCR(use_angle_cls=True, lang='en')
##########

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
######################

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()





######################

def table_detection(file_path):
    images = convert_from_path(file_path, dpi=800)
    first_page = images[0] # must be changed later to adapt for all pages
    image = first_page.convert("RGB")
    width, height = image.size
    #image.resize((int(width*0.5), int(height*0.5)))
    
    feature_extractor = DetrImageProcessor()
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    width, height = image.size
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'])
    return results['boxes'], image

#####################
ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print(f"ram usage : {ram_usage}")



#######################



def iterateFolder(root):
    count = 0
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        start_time = time.time()
        
        pred_bbox = table_detection(file_path)
        
        
        count += 1
        
        end_time = time.time()
        time_usage = end_time - start_time
        ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        print(f"Iteration {count + 1} - RAM Usage: {ram_usage:.2f} MB, Time Usage: {time_usage:.2f} seconds")

        if count > 2:
            break

def pdfFile(file_path):
    pred_bbox, image = table_detection(file_path)
    print(pred_bbox)
    #count += 1
    #ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return pred_bbox, image 

pdf_path = '/home/balbakri/ragger/first_model/CRM/578-2 Zert._englische Fassung_v3.pdf'
pred_bbox, image = table_detection(pdf_path)

bbox = np.array((pred_bbox[0].int()).tolist())
tol= 100
bbox = bbox + np.array([-tol,-tol,tol,tol])
cropped_image = image.crop(bbox)


####################################################
####################################################
####################################################
####################################################


from transformers import AutoImageProcessor, TableTransformerForObjectDetection
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
#feature_extractor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")

####################################################
'''
def cell_detection(file_path):

    image = Image.open(file_path).convert("RGB")
    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))


    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()

    with torch.no_grad():
      outputs = model(**encoding)


    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'])
    model.config.id2label
'''
def cell_detection(image):

    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()

    with torch.no_grad():
      outputs = model(**encoding)


    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'])
    model.config.id2label

cell_detection(cropped_image)


def compute_boxes(image):
    image = image.convert("RGB")
    width, height = image.size

    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    results = feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=[(height, width)])[0]
    boxes = results['boxes'].tolist()
    labels = results['labels'].tolist()

    return boxes,labels


def extract_text_from_image(image_np):
    # Initialize the PaddleOCR model
    ocr = PaddleOCR(use_angle_cls=False, lang='en',
                     #rec_model='ch_ppocr_server_v2.0_rec_infer',
                     rec_model_dir='/home/balbakri/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer'
                     ,return_word_box = False,
                     use_gpu = True,
                     save_crop_res=True
                     ,det_model_dir='/home/balbakri/.paddleocr/whl/det/en/ch_ppocr_server_v2.0_det_infer'
                     ,rec_char_dict_path='/home/balbakri/miniconda3/envs/dp/lib/python3.12/site-packages/paddleocr/ppocr/utils/custom2_dict.txt')

    # Perform OCR on the image
    result = ocr.ocr(image_np, cls=False)
    #print(result[0])
    ocr_results = result[0]
    # Extracting text components and combining them into a single string
    extracted_text = []
    y= []
    x = []
    for box in ocr_results:
        y.append(box[0][0][1])
        x.append(box[0][0][0])

    xy = ((np.array(x)/max(x))**2 + (np.array(y)/max(y))**2 )**.5
    sorted_indices_xy= [index for index, value in sorted(enumerate(xy), key=lambda x: x[1])]
    ocr_results_1 = [ocr_results[i] for i in sorted_indices_xy]
    print(ocr_results)
    color = (255, 0, 0) 
    print(ocr_results_1)
    # Line thickness of 2 px
    if len(ocr_results_1)>2:
        pts = np.array([[7261.0, 2092.0], [9238.0, 1863.0], [9399.0, 3123.0], [7422.0, 3352.0]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape to a 3D array for 'polylines'
        
        # Draw the polygon
        cv2.polylines(image_np, [pts], isClosed=True, color=(0,255,0), thickness=10)
        
        # Display the image
        plt.figure(figsize=(10, 7))  # Size of the figure in inches
        plt.imshow(image_np)
        plt.title('Image with Polygon')
        plt.axis('off')  # Hide the axes
        plt.show()

        #plt.show()
    for word_info in ocr_results_1:
        word = word_info[1][0]
        extracted_text.append(word)
        print(word)
    combined_text = ' '.join(extracted_text)
    return combined_text

def extract_table(image):
    image = image.convert("RGB")
    boxes,labels = compute_boxes(image)
    print('labels: ', labels)
    cell_locations = []

    for box_row, label_row in zip(boxes, labels):
        if label_row == 2:
            for box_col, label_col in zip(boxes, labels):
                if label_col == 1:
                    cell_box = (box_col[0], box_row[1], box_col[2], box_row[3])
                    cell_locations.append(cell_box)

    cell_locations.sort(key=lambda x: (x[1], x[0]))
    
    num_columns = 0
    box_old = cell_locations[0]

    for box in cell_locations[1:]:
        x1, y1, x2, y2 = box
        x1_old, y1_old, x2_old, y2_old = box_old
        num_columns += 1
        if y1 > y1_old:
            break
        
        box_old = box
        
    headers = []
    for box in cell_locations[:num_columns]:
        tol = .02
        enhancement = 8
        x1, y1, x2, y2 = box
        cell_image = image.crop((x1*(1-tol), y1*(1-tol), x2*(1+tol), y2*(1+tol))) 
        new_width = cell_image.width * enhancement
        new_height = cell_image.height * enhancement
        cell_image = cell_image.resize((new_width, new_height), resample=Image.LANCZOS)
        plt.imshow(cell_image)
        plt.show()
        custom_config = r'-l grc+eng --psm 1'
        #cell_text = pytesseract.image_to_string(cell_image,config=custom_config)
        cv2.imwrite('./image1.png', np.array(cell_image))

        image_np = np.array(cell_image)
        cell_text = extract_text_from_image(image_np)
        #print(cell_text)
        #image_file_like = image_to_filelike(cell_image)
        #elements = partition(file=image_file_like)
        #cell_text = elements[0].text
        headers.append(cell_text) 

    df = pd.DataFrame(columns=headers)

    row = []
    for box in cell_locations[num_columns:]:
        x1, y1, x2, y2 = box
        cell_image = image.crop((x1*(1-tol), y1*(1-tol), x2*(1+tol), y2*(1+tol))) 
        new_width = cell_image.width * enhancement
        new_height = cell_image.height * enhancement
        cell_image = cell_image.resize((new_width, new_height), resample=Image.LANCZOS)
        #cell_text = pytesseract.image_to_string(cell_image,config=custom_config)
        image_np = np.array(cell_image)
        cell_text = extract_text_from_image(image_np)
        #print(cell_text)

        
        plt.imshow(cell_image)
        plt.show()
        #if len(cell_text) > num_columns:
        #    cell_text = cell_text[:num_columns]

        row.append(cell_text)

        if len(row) == num_columns:
            df.loc[len(df)] = row
            row = []
            
    return df


df = extract_table(cropped_image)
df.to_csv('data.csv', index=False)
print(df)




import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(table=False, ocr=False, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
#img = cv2.imread(img_path)
result = table_engine(np.array(image))
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])
