from ultralytics import YOLO
import cv2
import pyttsx3

# Load YOLO model
prediction_model = YOLO("./model/model.pt")

# Define class names
class_names = ['bag', 'biscuits', 'bread', 'cake', 'caps', 'chips', 'chocolates', 'dal', 'deodorant', 'flattened_rice', 'flour', 'ice_cream', 'jelly', 'milk', 'mineral-water', 'noodles', 'oil', 'puffed_rice', 'rice', 'salt', 'semai', 'semolina', 'shoes', 'soft-drinks', 'soup', 'sugar', 'sunglass', 't-shirt', 'tea', 'watch']

# Initialize pyttsx3
engine = pyttsx3.init()

# Get all available voices
voices = engine.getProperty('voices')

# Set female voice
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Set speech rate (words per minute)
engine.setProperty('rate', 200)  # Adjust the rate as needed

# Load a single image
image_path = "C:/Users/HP/Documents/katha/test_images/7f8bcad7702a91d92c3552b3e9f9e0b3_large.png"
frame = cv2.imread(image_path)

# Perform inference using the YOLO model
results = prediction_model(frame)

# Iterate over the results
for result in results:
    boxes = result.boxes
    for box in boxes:
        conf = box.conf[0]
        cls = int(box.cls[0].item())  # Convert tensor to integer
        x1, y1, x2, y2 = box.xyxy[0]

        # Map class index to class name
        class_name = class_names[cls]

        # Draw bounding box and class name
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} -> Conf:{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Convert class name to audio message
        audio_message = f"{class_name} detected"
        engine.say(audio_message)
        engine.runAndWait()

# Display the annotated image
cv2.imshow("Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
