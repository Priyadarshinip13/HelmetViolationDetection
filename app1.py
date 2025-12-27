import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from dotenv import load_dotenv
import easyocr
import contextlib
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import os
import asyncio
import difflib
from pymongo import MongoClient
from dotenv import load_dotenv
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import warnings
from fpdf import FPDF
from email.mime.application import MIMEApplication

# Suppress warnings and logs
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(
    page_title="Helmet Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Cyber / Neon UI CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at top, #0f172a 0, #020617 45%, #000000 100%);
    color: #e5e7eb;
}

/* Main container padding */
.main {
    padding: 2rem 3rem;
}

/* Neon header */
.main-header {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(168,85,247,0.12));
    border: 1px solid rgba(56,189,248,0.5);
    box-shadow: 0 0 40px rgba(56,189,248,0.35);
    padding: 2rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: "";
    position: absolute;
    inset: -40%;
    background: conic-gradient(from 180deg at 50% 50%, #22d3ee, #a855f7, #f97316, #22d3ee);
    opacity: 0.18;
    filter: blur(32px);
    z-index: -1;
}

.main-header h1 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
    color: #e5e7eb;
}

.main-header p {
    font-size: 0.98rem;
    color: #9ca3af;
}

/* Glass cards */
.info-card, .challan-card, .metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(15,23,42,0.70));
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow: 0 18px 45px rgba(15,23,42,0.85);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.25s ease-out;
}

.info-card:hover, .challan-card:hover, .metric-card:hover {
    border-color: rgba(56,189,248,0.9);
    box-shadow: 0 0 35px rgba(56,189,248,0.45);
    transform: translateY(-2px);
}

/* Metric value/label */
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #22d3ee;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.8rem;
    color: #9ca3af;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Buttons */
.stButton>button {
    background: radial-gradient(circle at 0% 0%, #22d3ee, #a855f7 45%, #f97316 100%);
    color: #020617;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 2.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: 0 0 24px rgba(56,189,248,0.55);
}

.stButton>button:hover {
    filter: brightness(1.05);
    box-shadow: 0 0 32px rgba(168,85,247,0.75);
    transform: translateY(-1px);
}

/* Radio/file label tweaks */
.stRadio > label {
    font-weight: 600;
    font-size: 1.0rem;
    color: #e5e7eb;
    margin-bottom: 0.6rem;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.75rem;
    margin: 0.2rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-success {
    background: rgba(22, 163, 74, 0.18);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.35);
}

.status-warning {
    background: rgba(234, 179, 8, 0.16);
    color: #facc15;
    border: 1px solid rgba(250, 204, 21, 0.4);
}

.status-error {
    background: rgba(248, 113, 113, 0.16);
    color: #fca5a5;
    border: 1px solid rgba(248, 113, 113, 0.4);
}

/* Info / success / warning boxes */
.info-box, .success-box, .warning-box {
    border-radius: 12px;
    padding: 0.95rem 1.25rem;
    border: 1px solid rgba(148,163,184,0.4);
    background: radial-gradient(circle at top left, rgba(56,189,248,0.18), rgba(15,23,42,0.9));
    margin: 0.8rem 0;
}

.info-box p { color: #7dd3fc; font-weight: 500; margin: 0; }
.success-box p { color: #4ade80; font-weight: 500; margin: 0; }
.warning-box p { color: #facc15; font-weight: 500; margin: 0; }

/* Challan card specifics */
.challan-card {
    border-left: 3px solid rgba(250, 204, 21, 0.8);
}

.challan-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #facc15;
    margin-bottom: 0.8rem;
}

.challan-detail {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px dashed rgba(148,163,184,0.35);
    font-size: 0.92rem;
}

.challan-label {
    font-weight: 600;
    color: #e5e7eb;
}

.challan-value {
    color: #cbd5f5;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(15,23,42,0.85);
    border-radius: 999px;
    padding: 0.35rem 1.1rem;
    border: 1px solid rgba(51,65,85,0.9);
    color: #9ca3af;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-color: rgba(56,189,248,0.85);
    color: #e5e7eb;
    box-shadow: 0 0 18px rgba(56,189,248,0.45);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: radial-gradient(circle at top, #020617 0, #020617 45%, #000000 100%);
    border-right: 1px solid rgba(30,64,175,0.7);
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #22d3ee, #a855f7, transparent);
    margin: 2rem 0;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Helmet Violation Detection System</h1>
    <p>AI Helmet & Number Plate Recognition with Automated E-Challan Generation</p>
</div>
""", unsafe_allow_html=True)

# Fix for Windows asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load models
@st.cache_resource
def load_models():
    model = YOLO("C:/Users/PC-1/Desktop/project/best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

with st.spinner('🔄 Initializing detection engines...'):
    model, reader = load_models()

# Result file
os.makedirs("detected_numbers", exist_ok=True)
save_file = os.path.join("detected_numbers", "ocr_results.txt")
if not os.path.exists(save_file):
    with open(save_file, "w") as f:
        f.write("Timestamp\tPlate Number\n")

# MongoDB config
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["database"]
users_collection = db["user_details"]
challans_collection = db["challans"]

# Twilio config
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_PHONE")
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Email config
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_PASS = os.getenv("EMAIL_HOST_PASSWORD")

# Sidebar with model accuracy
with st.sidebar:
    st.markdown("### 📊 Model Performance")

    def get_accuracy_info():
        try:
            df = pd.read_csv("runs/detect/train29/results.csv")
            last_row = df.iloc[-1]
            return last_row['metrics/mAP_0.5'], last_row['metrics/mAP_0.5:0.95']
        except:
            return None, None

    map50, map5095 = get_accuracy_info()

    if map50:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">mAP@0.5</div>
                <div class="metric-value">{map50:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">mAP@0.5:0.95</div>
                <div class="metric-value">{map5095:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    st.info("""
**Features:**
- Real-time detection
- OCR recognition
- Auto E-challan
- SMS & Email alerts
""")

    st.markdown("---")
    st.markdown("### 📈 Statistics")
    try:
        total_challans = challans_collection.count_documents({})
        st.metric("Total Challans Issued", total_challans)
    except:
        st.metric("Total Challans Issued", "N/A")

# Detection function
def detect_and_display(frame, timestamp=None):
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model(frame)
    detected_numbers = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            margin = 20
            h, w, _ = frame.shape
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if label == 'number plate':
                cropped = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                dilated = cv2.dilate(gray, np.ones((2, 2), np.uint8), iterations=1)
                ocr_result = reader.readtext(
                    dilated,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    rotation_info=[0]
                )

                combined_plate = ""
                for (_, text, prob) in ocr_result:
                    text = text.strip().upper().replace(" ", "")
                    if prob > 0.10 and len(text) >= 1:
                        combined_plate += text

                if len(combined_plate) >= 3:
                    detected_numbers.append(combined_plate)
                    print(f"[INFO] Detected Number Plate: {combined_plate}")
                    cv2.putText(frame, f"Plate: {combined_plate}", (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    with open(save_file, "a") as f:
                        f.write(f"{timestamp or datetime.datetime.now()}\t{combined_plate}\n")

    return frame, detected_numbers

def generate_challan_pdf(challan_doc, user):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(200, 10, txt="Helmet Violation Detection System", ln=True, align='C')

    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(200, 10, txt="Helmet Violation Challan", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    data = [
        ("Challan No.", str(challan_doc['_id'])),
        ("Violation Date & Time", challan_doc['violation_datetime'].strftime('%d-%b-%Y %H:%M')),
        ("Vehicle Number", challan_doc['vehicle_no']),
        ("License Number", "N/A"),
        ("Violation", challan_doc['violation_type']),
        ("Fine Amount (Rs.)", str(challan_doc['fine_amount'])),
        ("Previous Dues (Rs.)", str(challan_doc['previous_fine_amount'])),
        ("Total Fine Due (Rs.)", str(challan_doc['total_fine_due'])),
        ("Location", challan_doc['location']['address']),
        ("Officer In Charge", challan_doc['officer_in_charge']),
        ("Status", challan_doc['challan_status']),
    ]
    for key, value in data:
        pdf.cell(60, 10, txt=f"{key}:", border=0)
        pdf.cell(100, 10, txt=value, border=0, ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Violation Evidence Image:", ln=True)

    if 'image_path' in challan_doc and os.path.exists(challan_doc['image_path']):
        pdf.image(challan_doc['image_path'], x=60, w=90)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 10, txt="(No image found)", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(200, 10, txt="Note: This is a system-generated challan. No signature required.", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)

    filepath = os.path.join("detected_numbers", f"challan_{challan_doc['_id']}.pdf")
    pdf.output(filepath)
    return filepath

# Email sender
def send_email(to_email, plate, fine_amount, previous_fine, total_due, challan_doc, user):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"🚨 Helmet Violation - {plate}"

        body = f"""
        <h2>Traffic Violation Notice</h2>
        <p><strong>Vehicle Number:</strong> {plate}</p>
        <p><strong>Violation:</strong> No Helmet</p>
        <p><strong>Fine:</strong> Rs.{fine_amount}</p>
        <p><strong>Previous Dues:</strong> Rs.{previous_fine}</p>
        <p><strong>Total Due:</strong> Rs.{total_due}</p>
        <p><strong>Location:</strong> MG Road, Bengaluru</p>
        <p><strong>Date & Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        msg.attach(MIMEText(body, 'html'))

        pdf_path = generate_challan_pdf(challan_doc, user)
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(part)

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Email error: {e}")


# SMS sender
def send_sms(to_phone, plate, fine_amount, previous_fine, total_due):
    try:
        if not to_phone.startswith('+91'):
            to_phone = '+91' + to_phone

        message = twilio_client.messages.create(
            body=(
                f"Traffic Violation (No Helmet)\n"
                f"Plate: {plate}\n"
                f"Fine: Rs.{fine_amount}, Prev: Rs.{previous_fine}, Total: Rs.{total_due}"
            ),
            from_=TWILIO_FROM,
            to=to_phone
        )
        print(f"SMS sent to {to_phone}: SID {message.sid}")
    except Exception as e:
        print(f"SMS error: {e}")

# Challan generator
def create_challan(plate, violation_frame=None):
    user = users_collection.find_one({"vehicle_no": plate})
    if not user:
        st.markdown(f"""
        <div class="warning-box">
            <p>⚠️ No user found for vehicle number: <strong>{plate}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        return

    user_id = user["_id"]
    now = datetime.datetime.now()
    start_of_day = datetime.datetime.combine(now, datetime.time.min)
    end_of_day = datetime.datetime.combine(now, datetime.time.max)

    count_today = challans_collection.count_documents({
        "user": user_id,
        "violation_datetime": {"$gte": start_of_day, "$lte": end_of_day}
    })
    if count_today >= 9000:
        st.markdown(f"""
        <div class="warning-box">
            <p>⚠️ Maximum 5 challans already issued today for <strong>{plate}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        return

    previous_challans = list(challans_collection.find({"user": user_id}))
    previous_fine = sum(ch["fine_amount"] for ch in previous_challans if not ch.get("is_paid", False))

    image_path = f"detected_numbers/violation_{plate}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    if violation_frame is not None:
        cv2.imwrite(image_path, violation_frame)

    challan_doc = {
        "user": user_id,
        "vehicle_no": user["vehicle_no"],
        "fine_amount": 500,
        "previous_fine_amount": previous_fine,
        "total_fine_due": previous_fine + 500,
        "violation_type": "No Helmet",
        "violation_datetime": now,
        "image_path": image_path,
        "location": {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "address": "Detected via CCTV at MG Road, Bengaluru"
        },
        "image_proof_url": "/images/sample-violation.jpg",
        "officer_in_charge": "Automated System",
        "challan_status": "Pending"
    }

    inserted = challans_collection.insert_one(challan_doc)
    created_challan = challans_collection.find_one({"_id": inserted.inserted_id})

    send_sms(user["phone_no"], plate, challan_doc["fine_amount"], challan_doc["previous_fine_amount"], challan_doc["total_fine_due"])
    send_email(user["email"], plate, challan_doc["fine_amount"],
               challan_doc["previous_fine_amount"], challan_doc["total_fine_due"],
               challan_doc, user)

    st.markdown("""
    <div class="success-box">
        <p>✅ Challan created and notification sent successfully!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="challan-card">
        <div class="challan-title">🚨 E-Challan Summary</div>
        <div class="challan-detail">
            <span class="challan-label">Name:</span>
            <span class="challan-value">{user.get("name", "N/A")}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Phone Number:</span>
            <span class="challan-value">{user.get("phone_no", "N/A")}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Vehicle Number:</span>
            <span class="challan-value">{created_challan['vehicle_no']}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Fine Amount:</span>
            <span class="challan-value">Rs.{created_challan['fine_amount']}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Previous Dues:</span>
            <span class="challan-value">Rs.{created_challan['previous_fine_amount']}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Total Fine Due:</span>
            <span class="challan-value"><strong>Rs.{created_challan['total_fine_due']}</strong></span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Violation:</span>
            <span class="challan-value">{created_challan['violation_type']}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Date & Time:</span>
            <span class="challan-value">{created_challan['violation_datetime'].strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Location:</span>
            <span class="challan-value">{created_challan['location']['address']}</span>
        </div>
        <div class="challan-detail">
            <span class="challan-label">Status:</span>
            <span class="challan-value"><span class="status-badge status-warning">{created_challan['challan_status']}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pdf_path = generate_challan_pdf(created_challan, user)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Challan PDF",
            data=f.read(),
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            key=f"download_{created_challan['_id']}"
        )

    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if 'image_path' in created_challan and os.path.exists(created_challan['image_path']):
            os.remove(created_challan['image_path'])
    except Exception as e:
        st.warning(f"⚠️ Could not delete temporary files: {e}")

# Main UI with tabs
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📡 Live Webcam"])

with tab1:
    st.markdown("### Upload an Image for Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_uploader")

    with col2:
        st.markdown("""
        <div class="info-box">
            <p><strong>Supported formats:</strong><br>JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        with st.spinner('🔍 Analyzing image...'):
            result_img, numbers = detect_and_display(frame)

        st.markdown("#### Detection Results")
        st.image(result_img, caption="Processed Image", width=400)

        if numbers:
            st.markdown(f"""
            <div class="success-box">
                <p>✅ Detected {len(numbers)} number plate(s): <strong>{', '.join(numbers)}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            for plate in numbers:
                create_challan(plate, frame)
        else:
            st.markdown("""
            <div class="info-box">
                <p>ℹ️ No violations detected in this image</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Upload a Video for Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="video_uploader")

    with col2:
        st.markdown("""
        <div class="info-box">
            <p><strong>Supported formats:</strong><br>MP4, AVI, MOV</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        plate_log = []
        prev_plate = ""
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame, numbers = detect_and_display(frame, timestamp)

            for num in numbers:
                if len(num) >= 5 and difflib.SequenceMatcher(None, num, prev_plate).ratio() < 0.85:
                    plate_log.append((timestamp, num))
                    prev_plate = num

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=500)
            progress = min(frame_count / max(total_frames, 1), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

        cap.release()
        progress_bar.empty()
        status_text.empty()

        if plate_log:
            st.markdown(f"""
            <div class="success-box">
                <p>✅ Analysis complete! Found {len(plate_log)} violation(s)</p>
            </div>
            """, unsafe_allow_html=True)
            for t, num in plate_log[:1]:
                st.write(f"**First Detected Plate:** {num}")
                create_challan(num, frame)
        else:
            st.markdown("""
            <div class="info-box">
                <p>ℹ️ No violations detected in this video</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Live Webcam Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        mobile_ip = st.text_input(
            "Mobile IP Webcam URL",
            value=os.getenv("MOBILE_IP", "http://192.168.1.11:8080/video"),
            help="Enter your mobile IP webcam URL"
        )

    with col2:
        st.markdown("""
        <div class="info-box">
            <p><strong>Setup:</strong><br>
            1. Install IP Webcam app<br>
            2. Start server<br>
            3. Enter URL above</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔴 Start Live Detection", key="start_webcam"):
        cap = cv2.VideoCapture(mobile_ip)

        if not cap.isOpened():
            st.markdown("""
            <div class="warning-box">
                <p>❌ Failed to connect. Please check:<br>
                • IP address is correct<br>
                • IP Webcam app is running<br>
                • Devices are on same network</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            stframe = st.empty()
            status_text = st.empty()
            stop_button = st.button("⏹️ Stop Detection", key="stop_webcam")

            plate_log = []
            prev_plate = ""
            frame_count = 0

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 10 == 0:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    processed_frame, numbers = detect_and_display(frame, timestamp)

                    for num in numbers:
                        if len(num) >= 5 and difflib.SequenceMatcher(None, num, prev_plate).ratio() < 0.85:
                            status_text.markdown(f"""
                            <div class="success-box">
                                <p>🚨 LIVE DETECTION: {timestamp} - Plate: <strong>{num}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            create_challan(num, frame)
                            prev_plate = num

                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)
                else:
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)

                frame_count += 1

            cap.release()
            st.success("✅ Live detection stopped")

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p><strong>Helmet Violation Detection System</strong> | AI by YOLOv8 & EasyOCR</p>
    <p style="font-size: 0.85rem;">© 2025 - Automated Traffic Enforcement</p>
</div>
""", unsafe_allow_html=True)
