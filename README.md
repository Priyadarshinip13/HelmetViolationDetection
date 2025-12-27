Helmet Violation Detection System using YOLOv8

An AI-based system to automatically detect helmet violations from traffic footage, recognize vehicle number plates, generate digital e-challans, and notify vehicle owners via SMS and email.

📌 Project Overview

This project implements an automated helmet violation detection system using YOLOv8, CNN with CBAM attention, and OCR. The system processes live CCTV feeds, recorded videos, or images to identify riders not wearing helmets. Once a violation is confirmed, it extracts the vehicle number plate, stores violation details in a database, generates an e-challan, and sends real-time notifications using Twilio SMS and Email.

The system is designed to reduce manual traffic monitoring, improve enforcement accuracy, and support smart traffic management initiatives.

Key Features:

Real-time helmet violation detection

Detection of rider, helmet / no-helmet, and number plate

OCR-based vehicle number extraction

Automatic e-challan generation

SMS notification using Twilio API

Email notification with challan PDF

MongoDB-based violation and user data storage

Web-based dashboard using Streamlit

Supports image, video, and live webcam input

🛠️ Technologies Used

Programming Language: Python

Object Detection: YOLOv8

Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/helmet-violation-detection.git
cd helmet-violation-detection

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables (.env)

Create a .env file and add:

MONGO_URI=mongodb://localhost:27017
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxx
TWILIO_TOKEN=xxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_PHONE=+1xxxxxxxxxx

EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password

MOBILE_IP=http://192.168.x.x:8080/video


⚠️ If using Twilio Trial, verify the recipient phone number in Twilio Console.

▶️ Run the Application
streamlit run app1.py


Open browser at:

http://localhost:8501

Deep Learning Framework: PyTorch

Computer Vision: OpenCV

OCR: EasyOCR

Database: MongoDB

Web Framework: Streamlit

SMS Service: Twilio API

Email Service: SMTP

UI: Streamlit with custom CSS
