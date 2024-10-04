# Facial-Expression-Detector
### Importing Required Libraries:
```python
from keras.models import load_model
```
- **`load_model`** is a function from the `keras.models` module used to load pre-trained models that were saved in a file (in this case, a face emotion detection model).
  
```python
from time import sleep
```
- **`sleep`** is a function from the `time` module used to pause the execution of the program for a specified duration. Though it's imported here, it is not used in this code.

```python
from keras.preprocessing.image import img_to_array
```
- **`img_to_array`** is used to convert an image to a NumPy array, which is required for the model to make predictions.

```python
from keras.preprocessing import image
```
- This imports the `image` module from `keras.preprocessing` that provides tools for image preprocessing, though it's not explicitly used in this code.

```python
import cv2
```
- **`cv2`** is the OpenCV library, which is used for real-time computer vision tasks, such as capturing video, detecting objects (like faces), and working with images.

```python
import numpy as np
```
- **`numpy`** (imported as `np`) is a library for numerical operations in Python, mainly used here to manipulate images and handle arrays.

### Loading Face Detection and Emotion Recognition Models:
```python
face_classifier = cv2.CascadeClassifier(r'C:\...\haarcascade_frontalface_default.xml')
```
- This loads a pre-trained face detection model (Haar Cascade), which is stored in the XML file. The model is used to detect faces in an image or video.

```python
classifier = load_model(r'C:\...\model.h5')
```
- This loads the pre-trained face emotion recognition model, which is stored in the `model.h5` file. This model will predict the emotion from a detected face.

### Emotion Labels:
```python
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
```
- This is a list of emotion labels corresponding to the emotions the model is trained to detect.

### Capturing Video Feed:
```python
cap = cv2.VideoCapture(0)
```
- This initializes video capture using the default camera (the argument `0` typically refers to the primary camera on your device).

### Loop for Continuous Video Frame Processing:
```python
while True:
    _, frame = cap.read()
```
- **`cap.read()`** reads each frame from the video feed. The variable `frame` holds the current frame captured by the camera. The underscore (`_`) is used for the first return value, which indicates if the frame was successfully read.

```python
    labels = []
```
- This initializes an empty list `labels` to hold the predicted emotion labels for detected faces (though this isn't used further in the code).

### Converting to Grayscale:
```python
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
- This converts the current frame from the default BGR (Blue-Green-Red) color format to grayscale, which is easier and faster to process for face detection.

### Detecting Faces:
```python
    faces = face_classifier.detectMultiScale(gray)
```
- **`detectMultiScale`** detects multiple faces in the grayscale image. It returns a list of rectangles where faces were detected. Each rectangle is defined by the `(x, y, w, h)` coordinates: the top-left corner (`x`, `y`), and the width (`w`) and height (`h`) of the rectangle (bounding box for each face).

### Drawing Rectangle and Preprocessing the Face Region:
```python
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
```
- For each face detected, a rectangle is drawn on the original frame. The `cv2.rectangle()` function takes the top-left `(x, y)` and bottom-right `(x+w, y+h)` coordinates, the rectangle color `(0, 255, 255)` (yellow), and the thickness of the rectangle border `2`.

```python
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
```
- **`roi_gray`** extracts the region of interest (ROI) from the grayscale image, which corresponds to the detected face.
- The ROI is resized to 48x48 pixels, the input size expected by the emotion detection model. **`INTER_AREA`** is an interpolation method used for resizing the image.

### Preparing Face ROI for Model Prediction:
```python
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
```
- **`np.sum([roi_gray]) != 0`** checks if the face region contains any information (non-zero pixels).
- **`astype('float') / 255.0`** normalizes the pixel values of the face ROI to a range between 0 and 1 (required for the model input).
- **`img_to_array(roi)`** converts the face ROI to a NumPy array.
- **`np.expand_dims(roi, axis=0)`** adds an extra dimension to the array so that it can be fed into the model (which expects a batch of images).

### Emotion Prediction:
```python
            prediction = classifier.predict(roi)[0]
```
- The pre-trained model predicts the emotion by processing the face ROI. **`classifier.predict(roi)`** returns an array of probabilities for each emotion class. **`[0]`** extracts the first prediction (since we're working with a single image).

```python
            label = emotion_labels[prediction.argmax()]
```
- **`prediction.argmax()`** returns the index of the highest probability in the prediction array, which corresponds to the detected emotion.
- **`emotion_labels[...]`** retrieves the corresponding emotion label from the `emotion_labels` list.

```python
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```
- **`label_position`** sets the coordinates where the emotion label will be displayed on the frame (top-left corner of the detected face).
- **`cv2.putText(...)`** draws the predicted emotion label on the frame with a green font (RGB value `(0, 255, 0)`) and thickness `2`.

### No Face Detected:
```python
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```
- If no face is detected in the current frame, the text **'No Faces'** is displayed at the coordinates `(30, 80)`.

### Displaying the Video Feed:
```python
    cv2.imshow('Emotion Detector', frame)
```
- **`cv2.imshow(...)`** displays the video feed in a window titled 'Emotion Detector' with the detected faces and emotion labels.

### Exiting the Loop:
```python
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
```
- **`cv2.waitKey(5)`** waits for a key press for 5 milliseconds. If the 'q' key is pressed, the loop breaks, and the program exits.

### Releasing Resources:
```python
cap.release()
cv2.destroyAllWindows()
```
- **`cap.release()`** releases the video capture object (stops the camera).
- **`cv2.destroyAllWindows()`** closes all OpenCV windows opened during execution.
