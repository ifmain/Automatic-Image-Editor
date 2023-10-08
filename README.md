# automatic_image_editor
This is a photo processing application that can automatically find objects and delete them or change them to others.

This application uses:
|Code Name|Usage|Link|
|--|--|--|
|YoloV8|Segmentation with segment names|[Link](https://github.com/ultralytics/ultralytics)|
|Stable Diffusion XL Inpainting|Remove and Replace Object|[Link](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)|
|VIT SWIN Base GPT2 Image Captioning|Сreation of image description|[Link](https://huggingface.co/Abdou/vit-swin-base-224-gpt2-image-captioning)|

**Many thanks to the developer of all models!**

## Exaple Usage

|Original|Raplaced|
|--|--|
|![bus](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)|![bus](ex/sportcar.jpg)|

Promts:
  Remove:Bus
  Replace:Sportcar

|Original|Raplaced|
|--|--|
|![bus](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)|![bus](ex/removed.png)|

Promts:
  Remove:Person
  Replace:

