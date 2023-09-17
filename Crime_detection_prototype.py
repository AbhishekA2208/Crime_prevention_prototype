import cv2

# Load the pre-trained model (you can use other pre-trained models as well)
model = cv2.dnn.readNet(r"D:\Dataset\yolov3.weights",r"D:\Dataset\yolov3.cfg")

# Load the class labels
with open("D:\Dataset\coco.data", "r") as file:
    classes = file.read().strip().split("\n")

# Initialize the video capture (you can use a camera or a video file)
cap = cv2.VideoCapture(r"D:\Dataset\bandicam 2023-09-13 09-13-47-541.mp4"")  # Replace with your video source

# Initialize a variable to keep track of suspicious activity
suspicious_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)

    # Get the detection results
    layer_names = model.getUnconnectedOutLayersNames()
    detections = model.forward(layer_names)

    # Process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                if class_id < len(classes):  # Check if class_id is within the valid range
                    detected_class = classes[class_id]
                    if detected_class == "suspicious":
                        suspicious_detected = True

# Release the video capture
cap.release()

# Print the result based on suspicious activity detection

if suspicious_detected:
    print("Suspicious activity detected.")
else:
    print("Normal activity.")