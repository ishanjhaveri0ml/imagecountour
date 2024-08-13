import os
from flask import Flask, request, render_template, jsonify
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) #used to create directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_marked_levels(image_path):

    image = cv2.imread(image_path)
    # the below code is used to detect edges, compress image , reduce noise (blurr) in the image .
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_levels = []

    for contour in contours:
        #This checks if the area of the current contour is less than 100. If it is, it means the contour is too small to be of interest
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # x & y are top left corner cordinates of the rectangle
        # (x+w, y+h)  are bottom right cordinates of rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #It takes the portion of gray from y to y+h in the vertical direction and from x to x+w in the horizontal direction
        marked_level = gray[y:y+h, x:x+w]
        marked_levels.append(marked_level)

    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "output_image.jpg"), image)

    return marked_levels

@app.route('/')
def upload_file():
   return render_template('index.html')

@app.route('/edit', methods=['POST'])
def upload_image():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(image_path)

        extracted_levels = extract_marked_levels(image_path)
        #to convert into the list
        return jsonify(levels=[level.tolist() for level in extracted_levels])
    else:
        return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)

