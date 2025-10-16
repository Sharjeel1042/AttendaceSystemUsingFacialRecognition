# Face Recognition Attendance System - Web Application

## Overview
This is a real-time face recognition attendance system with a web-based interface. It uses Flask-SocketIO for bidirectional communication between the server (PC) and client (phone/browser).

## Features
- **Real-time Face Recognition**: Capture and recognize faces instantly
- **Student Registration**: Register new students with webcam images
- **Attendance Tracking**: Automatic attendance marking with timestamp
- **View Records**: See attendance history
- **Mobile-Friendly**: Works on phones and tablets

## Project Structure
```
AttendaceSystemUsingFacialRecognition/
├── main/
│   ├── server.py              # Flask-SocketIO server
│   ├── main.py                # Desktop application (original)
│   ├── models.py
│   ├── database_utils.py
│   ├── registration.py
│   ├── faceDetection.py
│   ├── faceEmbedding.py
│   ├── recognition.py
│   └── real_time.py
├── client/
│   ├── templates/
│   │   └── index.html         # Web interface
│   └── static/
│       ├── css/
│       │   └── style.css      # Styling
│       └── js/
│           └── app.js         # Client-side JavaScript
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── dataset/                   # Student face images
├── database/
│   └── attendance.db         # SQLite database
└── requirements.txt
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Find your PC's IP address:**
- Windows: `ipconfig` (look for IPv4 Address)
- Linux/Mac: `ifconfig` or `ip addr show`

## Usage

### Running the Web Server

1. **Start the server:**
```bash
cd main
python server.py
```

2. **Access from your phone:**
- Make sure your phone and PC are on the **same WiFi network**
- Open your phone's browser
- Go to: `http://YOUR_PC_IP:5000`
- Example: `http://192.168.1.100:5000`

3. **Access from PC:**
- Open browser and go to: `http://localhost:5000`

### Using the Web Interface

#### Mark Attendance
1. Click "Mark Attendance" tab
2. Click "Start Camera"
3. Click "Capture & Recognize"
4. The system will recognize faces and mark attendance automatically

#### Register New Student
1. Click "Register Student" tab
2. Fill in student details (Reg No, Name, Semester, Phone)
3. Click "Start Camera"
4. Click "Capture Image" multiple times (at least 5 images)
5. Click "Register Student"

#### View Records
1. Click "View Records" tab
2. Click "Refresh Records" to see latest attendance

## Technical Details

### Server-Side (PC)
- **Flask**: Web framework
- **Flask-SocketIO**: Real-time bidirectional communication
- **OpenCV**: Face detection using DNN
- **keras-facenet**: Face recognition embeddings
- **SQLite**: Database for storing student info and attendance

### Client-Side (Phone/Browser)
- **Socket.IO**: Real-time communication with server
- **WebRTC**: Access device camera
- **HTML5 Canvas**: Capture and encode images
- **Responsive CSS**: Mobile-friendly design

### How It Works
1. **Client** captures image from phone camera
2. **Image** is encoded to base64 and sent via Socket.IO
3. **Server** receives image, detects faces, extracts embeddings
4. **Server** compares embeddings with database
5. **Server** sends recognition results back to client in real-time
6. **Client** displays results instantly

## Troubleshooting

### Camera not working
- Grant camera permissions in browser
- Use HTTPS or localhost (some browsers require secure connection)
- Try different browser (Chrome/Safari recommended)

### Cannot connect from phone
- Verify both devices are on same WiFi
- Check firewall settings (allow port 5000)
- Try disabling antivirus temporarily

### Recognition not accurate
- Capture more training images (8-10 recommended)
- Ensure good lighting
- Face should be clearly visible
- Avoid extreme angles

## Security Notes
For production use:
- Add authentication (login system)
- Use HTTPS instead of HTTP
- Add rate limiting
- Validate all inputs
- Use environment variables for sensitive data

## Running Original Desktop App
If you want to use the desktop version instead:
```bash
cd main
python main.py
```

## Support
For issues or questions, please check the code comments or refer to the documentation.

