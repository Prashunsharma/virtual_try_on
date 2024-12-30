from flask import Flask, render_template, Response, jsonify, request
import os
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)

# Shirt folder path (ensure you have shirt images in the static folder)
shirt_folder = 'static/shirt'
shirt_images = [f"{shirt_folder}/{filename}" for filename in os.listdir(shirt_folder) if filename.endswith('.png')]

# Initialize camera and pose detector
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Default shirt index
shirt_index = 0
img_Shirt = cv2.imread(shirt_images[shirt_index], cv2.IMREAD_UNCHANGED)
fixed_ratio = img_Shirt.shape[0] / img_Shirt.shape[1]  # Height / Width


def gen_frames():
    global shirt_index, img_Shirt, fixed_ratio
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (960, 720))

        # Detect pose and get landmarks
        img = detector.findPose(img, draw=False)
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        if lmList:
            # Get relevant landmarks: shoulders and hips
            lm11 = lmList[11][:2]  # Left shoulder
            lm12 = lmList[12][:2]  # Right shoulder
            lm23 = lmList[23][:2]  # Left hip
            lm24 = lmList[24][:2]  # Right hip

            # Calculate shirt width based on shoulder distance
            shirt_width = int(abs(lm12[0] - lm11[0])*1.9)  # Scaled width
            shirt_height = int(shirt_width * fixed_ratio*1.15)  # Maintain aspect ratio

            if shirt_width > 0 and shirt_height > 0:
                resized_shirt = cv2.resize(img_Shirt, (shirt_width, shirt_height))

                # Calculate top-left position for the shirt
                shoulder_center_x = (lm11[0] + lm12[0]) // 2
                shoulder_center_y = (lm11[1] + lm12[1]) // 2
                hip_center_y = (lm23[1] + lm24[1]) // 2  # Average y-coordinate of hips
                chest_center_y = shoulder_center_y + (hip_center_y - shoulder_center_y) // 3  # Position shirt on chest
                
                top_left_x = shoulder_center_x - shirt_width // 2
                top_left_y = chest_center_y - shirt_height // 2  + int(-0.27* shirt_height) # Adjust to center the shirt on the chest

                # Overlay the shirt image on the frame
                try:
                    img = cvzone.overlayPNG(img, resized_shirt, [top_left_x, top_left_y])
                except Exception as e:
                    print(f"Error overlaying shirt: {e}")
            else:
                print("Invalid dimensions for resizing.")
        else:
            print("Pose not detected. Ensure the subject is visible in the frame.")

        # Encode the image as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        img_str = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_str + b'\r\n\r\n')


@app.route('/')
def index():
    global shirt_index, img_Shirt
    img_Shirt = cv2.imread(shirt_images[shirt_index], cv2.IMREAD_UNCHANGED)
    return render_template('index.html', shirt_image=shirt_images[shirt_index])


@app.route('/change-shirt', methods=['POST'])
def change_shirt():
    global shirt_index, img_Shirt, fixed_ratio

    direction = request.json.get('direction')

    if direction == 'next':
        shirt_index = (shirt_index + 1) % len(shirt_images)
    elif direction == 'prev':
        shirt_index = (shirt_index - 1) % len(shirt_images)

    # Reload the shirt image
    img_Shirt = cv2.imread(shirt_images[shirt_index], cv2.IMREAD_UNCHANGED)
    fixed_ratio = img_Shirt.shape[0] / img_Shirt.shape[1]  # Update the aspect ratio

    return jsonify({"shirt_image": shirt_images[shirt_index]})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
