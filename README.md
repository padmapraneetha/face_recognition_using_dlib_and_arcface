# face_recognition_using_dlib_and_arcface
Face Recognition System with dlib and ArcFace
This project implements a face recognition system using two approaches: dlib and ArcFace. Both systems allow users to register faces, recognize faces from images or camera input, and process videos, all through a user-friendly Streamlit web interface.xtended to larger datasets.

Features:

Register Face: Capture or upload an image to register a person's face with a name.
Recognize from Camera: Capture a photo via webcam and identify faces in real-time.
Recognize from Upload: Upload images or videos to detect and recognize faces.
View Dataset: Display and manage registered faces, with options to delete individual entries or clear all.
Output: Displays recognized faces with bounding boxes, names, confidence scores, and processing time (in milliseconds).

Prerequisites:

Operating System: Windows, macOS, or Linux.
Python: Version 3.7–3.10 (Python 3.12 may have compatibility issues with dlib).
Webcam (optional): For real-time face recognition via camera.
Git: To clone the repository.


Installation

Clone the Repository
git clone <repository-url>
cd <repository-name>


Set Up a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install DependenciesInstall the required Python packages:
pip install streamlit opencv-python numpy pillow insightface face_recognition


