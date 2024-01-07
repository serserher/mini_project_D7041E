import os
import shutil
import xml.etree.ElementTree as ET


BASE_PATH = "VOC2012"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "JPEGimages"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "Annotations"])
IMAGES_PATH_REDUCED = os.path.sep.join([BASE_PATH, "Images_Reduced"])
ANNOTS_PATH_REDUCED = os.path.sep.join([BASE_PATH, "Annotations_Reduced"])

def has_single_bounding_box(object_element):
    # Check if an object has only one bounding box (ignoring parts)
    return len(object_element.findall('part')) == 0

def filter_annotations(input_folder_annotations, output_folder_annotations, input_folder_images, output_folder_images, limit_number = 10000):
    os.makedirs(output_folder_annotations, exist_ok=True)
    os.makedirs(output_folder_images, exist_ok=True)
    count = 0
    for filename in os.listdir(input_folder_annotations):
        if count >= limit_number:
            continue
        if filename.endswith(".xml"):
            xml_path = os.path.join(input_folder_annotations, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Check if the annotation has only one object with a single bounding box
            objects = root.findall('.//object')
            if len(objects) == 1 and has_single_bounding_box(objects[0]):
                output_path = os.path.join(output_folder_annotations, filename)
                # Copy the XML file to the output folder
                os.makedirs(output_folder_annotations, exist_ok=True)
                with open(output_path, 'wb') as output_file:
                    output_file.write(ET.tostring(root))
                   # Copy the corresponding JPEG image to the "JPEGImages" folder
                image_filename = os.path.splitext(filename)[0] + ".jpg"
                image_path_src = os.path.join(input_folder_images, image_filename)
                image_path_dest = os.path.join(output_folder_images, image_filename)
                shutil.copy(image_path_src, image_path_dest)
                count += 1 
input_folder_a = ANNOTS_PATH
output_folder_a = ANNOTS_PATH_REDUCED

input_folder_i = IMAGES_PATH
output_folder_i = IMAGES_PATH_REDUCED

filter_annotations(input_folder_a, output_folder_a, input_folder_i, output_folder_i, limit_number = 4000)


IMAGES_PATH_REDUCED_TEST = os.path.sep.join([BASE_PATH, "Images_Reduced_Test"])
ANNOTS_PATH_REDUCED_TEST = os.path.sep.join([BASE_PATH, "Annotations_Reduced_Test"])

output_folder_a = ANNOTS_PATH_REDUCED_TEST
output_folder_i = IMAGES_PATH_REDUCED_TEST

