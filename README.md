# Chaos_Analyze
---
## Team Name: Jose James Team
## Team Memebere: Jose James - Viswajyothi College of Engineering and Technology

## Project Description
- Like its name, it is a chaos analyzer that judges a room or environment based MIT level chaos metrics to say how messy it thinks the room is.


## The Problem
- It is only fair that people need to use an ai tool to judge how messy their surroundings are.
- Its just too difficult and too much of a work to look at your surroundings and make the judgement.

## The Solution
- *So*, the solution is simple. Just make the AI do the work.
-  You can just upload the image or take one directly from the page.
-  Next ask it to analyze.
-  It will give a percentage rating, a comment on how messy the room is, and then give bunch of technical jargon that you may not understand.
-  Also there are different views at the bottom—annotation, edges, and heatmap

---
## Technical Details

### Technologies Used
- **Written in**
  - Python
  - HTML/CSS (in st.markdown())
- ** Frameworks used**
  - Streamlit
  - OpenCV
  - YOLOv8
  - scikit-image
- ** Libraries Used**
  - streamlit                 # Web app framework
  - ultralytics              # YOLOv8 object detection
  - opencv-python-headless   # Computer vision (headless for deployment)
  - scikit-image            # Advanced image processing
  - numpy                   # Numerical computing
  - pillow                  # Image handling
- ** Tools Used**
  - **Development** : Git/Github, VScode, pip
  - **Deployment** : Streamlit community cloud
  - **Computer Vision Tools**: Canny Edge Detection (via OpenCV), Gaussian Blur (via OpenCV), Color Space Conversion (BGR/RGB/Lab/YUV), Heatmap Generation (cv2.COLORMAP_JET)
  - **AI/ML**: YOLO model, object detection pipeline

## Implementation
### Installation
- git clone https://github.com/Jomezh/Uselessday
- cd chaos-analyzer-pro
- pip install -r requirements.txt

### Run
streamlit run chaos_analyzer.py

## Project Documentation

### Screenshots
- <img src="Screenshot 2025-08-09 072616.png" />
- Describes the Overview of the web page on arrival
- <img width="3188" height="1202" alt="frame (3)" src="Screenshot 2025-08-09 072801.png" />
- Describes the result
- <img width="3188" height="1202" alt="frame (3)" src="Screenshot 2025-08-09 072837.png" />
-  Describes the result (bottom part)

## Project Demo
### Video
 [Demo Video](https://drive.google.com/file/d/1zKEKxQ-xcaY9-gC-7bnZqHxQgqsyM1R_/view?usp=sharing)

### Additional Demo
Website [Chaos-analyze](https://chaos-analyze.streamlit.app/#chaos-analyzer)

## Team Contributions
### Jose James (Solo)



