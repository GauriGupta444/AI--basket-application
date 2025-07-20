import cv2
from ultralytics import YOLO
import qrcode
import numpy as np

# 1. Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Define items and prices (expanded list)
item_prices = {
    # Fruits
    "banana": 10,
    "apple": 15,
    "orange": 12,
    "grape": 40,
    "kiwi": 25,
    
    # Vegetables
    "carrot": 20,
    "broccoli": 30,
    "tomato": 25,
    "potato": 15,
    "onion": 18,
    
    # Grocery items
    "bottle": 50,
    "wine glass": 120,
    "cup": 35,
    "fork": 15,
    "knife": 20,
    "spoon": 15,
    "bowl": 45,
    
    # Packaged goods
    "sandwich": 30,
    "hot dog": 25,
    "pizza": 90,
    "donut": 20,
    "cake": 150,
    
    # Dairy
    "milk": 55,
    "cheese": 80,
    "egg": 8,
    
    # Other
    "book": 100,
    "scissors": 40,
    "teddy bear": 200,
    "hair drier": 350,
    "toothbrush": 30
}

# 3. Open webcam
cap = cv2.VideoCapture(0)
detected_items = []

print("Press 'q' to generate Paytm payment QR...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    detected_items_current = []
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        
        if label in item_prices:
            detected_items_current.append(label)
            cv2.putText(frame, f"{label}: ₹{item_prices[label]}", 
                       (10, 30 + 30 * len(detected_items_current)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("AI Smart Basket", frame)
    
    # Press 'q' to generate Paytm QR
    if cv2.waitKey(1) == ord('q'):
        detected_items = detected_items_current
        if detected_items:
            total = sum(item_prices[item] for item in detected_items)
            
            # Create Paytm Deep Link
            upi_id = "8076747293@ptsbi"  # Replace with actual UPI ID
            paytm_link = f"upi://pay?pa={upi_id}&pn=Supermarket&am={total}&cu=INR"
            
            # Generate QR
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(paytm_link)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_cv = cv2.cvtColor(np.array(qr_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            # Show QR
            cv2.imshow("Scan to Pay via Paytm", qr_cv)
            print(f"Detected Items: {', '.join(detected_items)}")
            print(f"Total: ₹{total} | Scan QR to open Paytm!")
            
            # Hold window until ESC pressed
            while True:
                if cv2.waitKey(1) == 27:  # 27 = ESC key
                    cv2.destroyWindow("Scan to Pay via Paytm")
                    break
        else:
            print("No items detected!")
        break

cap.release()
cv2.destroyAllWindows()