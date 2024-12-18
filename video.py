# from ultralytics import YOLO
# import cv2

# # Load YOLO model
# prediction_model = YOLO("./model/model.pt")

# # Define class names
# class_names = ['bag', 'biscuits', 'bread', 'cake', 'caps', 'chips', 'chocolates', 'dal', 'deodorant', 'flattened_rice', 'flour', 'ice_cream', 'jelly', 'milk', 'mineral-water', 'noodles', 'oil', 'puffed_rice', 'rice', 'salt', 'semai', 'semolina', 'shoes', 'soft-drinks', 'soup', 'sugar', 'sunglass', 't-shirt', 'tea', 'watch']

# # Open laptop camera
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference using the YOLO model
#     results = prediction_model(frame)

#     # Iterate over the results
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             conf = box.conf[0]
#             cls = int(box.cls[0].item())  # Convert tensor to integer
#             x1, y1, x2, y2 = box.xyxy[0]

#             # Map class index to class name
#             class_name = class_names[cls]

#             # Draw bounding box and class name
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # Display the annotated image
#     cv2.imshow("Detection", frame)

#     # Break the loop when 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import pyttsx3

# Load YOLO model
prediction_model = YOLO("./model/model.pt")

# Define class names
class_names = ['bag', 'biscuits', 'bread', 'cake', 'caps', 'chips', 'chocolates', 'dal', 'deodorant', 'flattened_rice', 'flour', 'ice_cream', 'jelly', 'milk', 'mineral-water', 'noodles', 'oil', 'puffed_rice', 'rice', 'salt', 'semai', 'semolina', 'shoes', 'soft-drinks', 'soup', 'sugar', 'sunglass', 't-shirt', 'tea', 'watch']

# Initialize pyttsx3
engine = pyttsx3.init()

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

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

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

