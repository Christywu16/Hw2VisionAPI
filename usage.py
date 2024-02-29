import io
import os
import pandas as pd
from google.cloud import vision
from google_vision_ai import VisionAI
from google_vision_ai import prepare_image_local

# instance a client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_file_ai_vision_demo.json'
client = vision.ImageAnnotatorClient()

# # local source
# image_path = './image/img1.jpg'
# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()
# image =vision.Image(content=content)

# # link source
# image_url = 'https://res.cloudinary.com/traveltripperweb/image/upload/c_fit,f_auto,h_992,q_auto,w_992/v1647485242/nmdwm76leiauc2ubo6ct.jpg'
# image = vision.Image()
# image.source.image_uri = image_url

# response = client.label_detection(image=image)
# for label in response.label_annotations:
#     print(label.description)
#     print(label.score)


# labels df
image_path = './image/img5.jpg'
image  = prepare_image_local(image_path)
va = VisionAI(client, image)
label_detections = va.label_detection()

df = pd.DataFrame(label_detections)
print(df)

# # object detection
# image_path = './image/img7.jpg'
# image  = prepare_image_local(image_path)
# va = VisionAI(client, image)
# object_detections = va.object_detection()
# for object in object_detections:
#     print(object.name)
#     print(object.score)
