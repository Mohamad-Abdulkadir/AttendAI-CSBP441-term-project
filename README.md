
# AttendAI: Automated Attendance Tracking System

AttendAI is a real-time, AI-powered attendance tracking system designed for classroom environments. It uses a pre-trained YOLOv8 model for face detection and a FaceNet model for face recognition to automate and streamline the process of tracking attendance.

---

## Installation

To run AttendAI locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Clone the Repository
```bash
git clone https://github.com/Mohamad-Abdulkadir/AttendAI-CSBP441-term-project.git
```

### Install Dependencies
Run the following command to install all required libraries:
```bash
pip install streamlit ultralytics opencv-python-headless deepface numpy pandas gdown
```

---

## Usage

### Start the Streamlit App
Run the following command in your terminal:
```bash
streamlit run app.py
```

---

## Features
- Generate a database from student images stored in a Google Drive folder.
- Check attendance by uploading classroom images.
- Download attendance reports in CSV format.

---
