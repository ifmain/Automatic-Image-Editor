import time
import cv2
from diffusers import AutoPipelineForInpainting
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO
import gradio as gr
from gradio import components
import difflib

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load

def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_most_similar_string(target_string, string_array):
    differ = difflib.Differ()
    best_match = string_array[0]
    best_match_ratio = 0
    for candidate_string in string_array:
        similarity_ratio = difflib.SequenceMatcher(None, target_string, candidate_string).ratio()
        if similarity_ratio > best_match_ratio:
            best_match = candidate_string
            best_match_ratio = similarity_ratio
    
    return best_match

def loadModels():

    yoloModel=YOLO('yolov8x-seg.pt')
    pipe =AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        ).to("cuda")
    image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning", device=DEVICE)
    #return gpt_model, gpt_tokenizer, gpt_params,yoloModel,pipe,image_captioner
    return yoloModel,pipe,image_captioner

# Yolo

def getClasses(model,img1):
    results = model([img1])
    out=[]
    for r in results:
        #im_array = r.plot(boxes=False,labels=False)  # plot a BGR numpy array of predictions
        im_array = r.plot()
        out.append(r)

    return r,im_array[..., ::-1],results

def getMasks(out):
    allout={}
    class_masks = {}
    for a in out:
        class_name = a['name']
        mask = a['img']
        if class_name in class_masks:
            class_masks[class_name] = Image.fromarray(
                np.maximum(np.array(class_masks[class_name]), np.array(mask))
            )
        else:
            class_masks[class_name] = mask
    for class_name, mask in class_masks.items():
        allout[class_name]=mask
    return allout

def joinClasses(classes):
    i = 0
    out = []
    for r in classes:
        masks = r.masks
        name0 = r.names[int(r.boxes.cls.cpu().numpy()[0])]

        mask1 = masks[0]
        mask = mask1.data[0].cpu().numpy()

        # Normalize the mask values to 0-255 if needed
        mask_normalized = ((mask - mask.min()) * (255 / (mask.max() - mask.min()))).astype(np.uint8)

        # Add white border
        kernel = np.ones((10, 10), np.uint8)
        mask_with_border = cv2.dilate(mask_normalized, kernel, iterations=1)

        mask_img = Image.fromarray(mask_with_border, "L")
        out.append({'name': name0, 'img': mask_img})
        i += 1

    allMask = getMasks(out)
    return allMask

def getSegments(yoloModel,img1):
    classes,image,results1=getClasses(yoloModel,img1)
    im = Image.fromarray(image)  # RGB PIL image
    im.save('classes.jpg')
    allMask=joinClasses(classes)
    return allMask

# Gradio UI

def getDescript(image_captioner,img1):
    base64_img = image_to_base64(img1)
    caption = image_captioner(base64_img)[0]['generated_text']
    return caption

def rmGPT(caption,remove_class,change):
    arstr=caption.split(' ')
    popular=get_most_similar_string(remove_class,arstr)
    ind=arstr.index(popular)
    if len(change)<3:
        new=[]
        rng=round(len(arstr)/5)
        print(f'Center {ind} | range {ind-rng}:{ind+rng+1}')
        for i in range(len(arstr)):
            if i not in list(range(ind-rng,ind+rng)):
                new.append(arstr[i])
        return ' '.join(new)
    else:
        arstr[ind]=change
        return ' '.join(arstr)

# SDXL

def ChangeOBJ(sdxl_m,img1,response,mask1):
    size = img1.size
    image = sdxl_m(prompt=response, image=img1, mask_image=mask1).images[0]
    return image.resize((size[0], size[1]))



yoloModel,sdxl,image_captioner=loadModels()

def full_pipeline(image, target,change):
    img1 = Image.fromarray(image.astype('uint8'), 'RGB')
    allMask=getSegments(yoloModel,img1)
    tartget_to_remove=get_most_similar_string(target,list(allMask.keys()))
    caption=getDescript(image_captioner,img1)

    response=rmGPT(caption,tartget_to_remove,change)
    mask1=allMask[tartget_to_remove]

    remimg=ChangeOBJ(sdxl,img1,response,mask1)

    return remimg,caption,response



iface = gr.Interface(
    fn=full_pipeline, 
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Textbox(label="What to delete?"),
        gr.Textbox(label="Change?"),
    ], 
    outputs=[
        gr.Image(label="Result Image", type="numpy"),
        gr.Textbox(label="Caption"),
        gr.Textbox(label="Message"),
    ],
    live=False
)


#iface.launch(share=True)
iface.launch(server_name='192.168.31.75')