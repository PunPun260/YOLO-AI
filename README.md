# YOLO 001 : Training

YOLO is a python library that run AI model that are *for processing the image*. so How do where does the AI learn the data. That’s right you need to train it !!!

**Prepare (linux) :** 
1. first you gotta ***create the environment and get the libary***
	1. create the directory; 
	   `$ mkdir yolo` # anyname u want
	   `$ cd yolo`
	2. create the environment;
	   `$ python3 -m venv --system-site-package yolo_venv`
	3. get into the environment:
	   `$ source yolo_venv/bin/activate`
	4. get the library;
	   `(yolo_venv)$ pip install ultralytics ncnn`
2. Next we gotta label the data:
   <<< up coming >>>
3. Now u r ready!!

***

**Training :
1. you gotta create new ***python*** file name ‘train.py’ and put this code;
> 📄 train.py
```python 
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8s.pt")

# Train the model using your exported dataset
results = model.train(data="<yaml file address>", epochs=100, imgsz=640)
```

> [!tip]
epochs = how many times our model train

2. next run it;
`(yolo_venv)$ pyhton train.py`

3. now at the end it'll look like there’re some address. Congratulation!!, that’s your final model.

***

Testing :
1. you gotta create new ***python*** file name ‘test.py’ and put this code;
> 📄 test.py
```python 
from ultralytics import YOLO

# Load the trained model
model = YOLO("<final model adress>")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
	    break
	
    # Run YOLOv8 inference
    results = model(frame)
	
    # Draw results on frame
    annotated_frame = results[0].plot()
	
    # Show the frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
	
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
```

2. next run it;
`(yolo_venv)$ pyhton test.py`

3. Congratulation, now u can enjoy with your AI :)
