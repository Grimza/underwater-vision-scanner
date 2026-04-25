# Underwater Vision Scanner

This project detects underwater objects using a YOLO model trained on labeled images.

The system can identify the following classes:
- Car  
- Wheel  
- Unknown body  
- Tag  
- Logo  

## Installation
Install the required packages listed in *requirements.txt* (if you don't have them already) by running: `pip install -r requirements.txt`

Make sure you have Python properly installed and configured.

## How to Use
You can test the model with:

- An image: `python predict_image.py`  
- A video: `python predict_video.py`  

The model will output detected objects with bounding boxes and labels.

![example](https://raw.githubusercontent.com/Grimza/underwater-vision-scanner/5619a181a7d069da2c1306a509880e228c7169aa/assets/readme_example.webp) 

---
Note: If the image window does not open when running the script, it may be due to your system’s default image viewer. Some applications (such as Paint 3D on Windows) may not open images correctly when called from Python.

In that case, try changing the default image viewer to a different application (for example, the Windows Photos app).

Alternatively, you can save the output image instead of displaying it by adding the following line to your *predict_image.py* code:

`results[0].save(filename="result.png")`
