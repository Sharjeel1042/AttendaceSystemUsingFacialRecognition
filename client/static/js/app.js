// Initialize Socket.IO connection
const socket = io();

// DOM elements
const statusDot = document.querySelector('.dot');
const statusText = document.getElementById('status-text');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('start-camera');
const captureRecognizeBtn = document.getElementById('capture-recognize');
const stopCameraBtn = document.getElementById('stop-camera');
const recognitionResult = document.getElementById('recognition-result');

// Registration elements
const registerVideo = document.getElementById('register-video');
const registerCanvas = document.getElementById('register-canvas');
const startRegisterCameraBtn = document.getElementById('start-register-camera');
const captureImageBtn = document.getElementById('capture-image');
const clearImagesBtn = document.getElementById('clear-images');
const capturedImagesDiv = document.getElementById('captured-images');
const imageCountSpan = document.getElementById('image-count');
const submitRegistrationBtn = document.getElementById('submit-registration');
const registrationForm = document.getElementById('registration-form');
const registrationResult = document.getElementById('registration-result');

// Records elements
const refreshRecordsBtn = document.getElementById('refresh-records');
const attendanceRecords = document.getElementById('attendance-records');

let stream = null;
let registerStream = null;
let capturedImages = [];

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    statusDot.classList.add('connected');
    statusText.textContent = 'Connected';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Disconnected';
});

socket.on('connection_response', (data) => {
    console.log('Server response:', data.message);
});

socket.on('recognition_success', (data) => {
    displayRecognitionResult(data, true);
});

socket.on('recognition_result', (data) => {
    displayRecognitionResult(data, false);
});

socket.on('recognition_error', (data) => {
    showError(recognitionResult, data.error);
});

socket.on('registration_success', (data) => {
    showSuccess(registrationResult, data.message);
    registrationForm.reset();
    clearCapturedImages();

    // Stop registration camera
    if (registerStream) {
        registerStream.getTracks().forEach(track => track.stop());
        registerVideo.srcObject = null;
        startRegisterCameraBtn.disabled = false;
        captureImageBtn.disabled = true;
        clearImagesBtn.disabled = true;
    }
});

socket.on('registration_error', (data) => {
    showError(registrationResult, data.error);
});

socket.on('attendance_data', (data) => {
    displayAttendanceRecords(data.records);
});

// Tab switching
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');

    tabs.forEach(tab => tab.classList.remove('active'));
    buttons.forEach(btn => btn.classList.remove('active'));

    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    if (tabName === 'records') {
        loadAttendanceRecords();
    }
}

// Camera functions for recognition
startCameraBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        video.srcObject = stream;
        startCameraBtn.disabled = true;
        captureRecognizeBtn.disabled = false;
        stopCameraBtn.disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError(recognitionResult, 'Could not access camera. Please check permissions.');
    }
});

captureRecognizeBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    recognitionResult.innerHTML = '<p>🔍 Processing... Please wait</p>';
    recognitionResult.className = 'result-box info';
    recognitionResult.style.display = 'block';

    captureRecognizeBtn.disabled = true;

    socket.emit('recognize_face', { image: imageData });

    // Re-enable button after 2 seconds
    setTimeout(() => {
        captureRecognizeBtn.disabled = false;
    }, 2000);
});

stopCameraBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startCameraBtn.disabled = false;
        captureRecognizeBtn.disabled = true;
        stopCameraBtn.disabled = true;
    }
});

// Camera functions for registration
startRegisterCameraBtn.addEventListener('click', async () => {
    try {
        registerStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        registerVideo.srcObject = registerStream;
        startRegisterCameraBtn.disabled = true;
        captureImageBtn.disabled = false;
        clearImagesBtn.disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError(registrationResult, 'Could not access camera. Please check permissions.');
    }
});

captureImageBtn.addEventListener('click', () => {
    if (capturedImages.length >= 10) {
        showError(registrationResult, 'Maximum 10 images allowed');
        return;
    }

    const context = registerCanvas.getContext('2d');
    registerCanvas.width = registerVideo.videoWidth;
    registerCanvas.height = registerVideo.videoHeight;
    context.drawImage(registerVideo, 0, 0);

    const imageData = registerCanvas.toDataURL('image/jpeg', 0.8);
    capturedImages.push(imageData);

    const img = document.createElement('img');
    img.src = imageData;
    img.className = 'preview-image';
    capturedImagesDiv.appendChild(img);

    imageCountSpan.textContent = capturedImages.length;

    if (capturedImages.length >= 5) {
        submitRegistrationBtn.disabled = false;
    }

    // Visual feedback
    captureImageBtn.style.transform = 'scale(0.95)';
    setTimeout(() => {
        captureImageBtn.style.transform = 'scale(1)';
    }, 100);
});

clearImagesBtn.addEventListener('click', clearCapturedImages);

function clearCapturedImages() {
    capturedImages = [];
    capturedImagesDiv.innerHTML = '';
    imageCountSpan.textContent = '0';
    submitRegistrationBtn.disabled = true;
    registrationResult.style.display = 'none';
}

// Registration form submission
registrationForm.addEventListener('submit', (e) => {
    e.preventDefault();

    const regNo = document.getElementById('reg-no').value.trim();
    const name = document.getElementById('name').value.trim();
    const semester = document.getElementById('semester').value.trim();
    const phone = document.getElementById('phone').value.trim();

    if (capturedImages.length < 5) {
        showError(registrationResult, 'Please capture at least 5 images');
        return;
    }

    registrationResult.innerHTML = '<p>⏳ Processing registration... Please wait</p>';
    registrationResult.className = 'result-box info';
    registrationResult.style.display = 'block';

    submitRegistrationBtn.disabled = true;

    socket.emit('register_student', {
        reg_no: regNo,
        name: name,
        semester: semester,
        phone: phone,
        images: capturedImages
    });

    // Re-enable button after 5 seconds (in case of timeout)
    setTimeout(() => {
        submitRegistrationBtn.disabled = false;
    }, 5000);
});

// Attendance records
refreshRecordsBtn.addEventListener('click', loadAttendanceRecords);

function loadAttendanceRecords() {
    attendanceRecords.innerHTML = '<p>⏳ Loading records...</p>';
    socket.emit('get_attendance', {});
}

function displayAttendanceRecords(records) {
    if (records.length === 0) {
        attendanceRecords.innerHTML = '<p style="text-align: center; color: #718096; padding: 20px;">No attendance records found.</p>';
        return;
    }

    attendanceRecords.innerHTML = '';
    records.forEach(record => {
        const div = document.createElement('div');
        div.className = 'record-item';
        div.innerHTML = `
            <div class="record-info">
                <strong>${record.name}</strong> (${record.reg_no})
            </div>
            <div class="record-time">📅 ${record.timestamp}</div>
        `;
        attendanceRecords.appendChild(div);
    });
}

// Display functions
function displayRecognitionResult(data, isSuccess) {
    if (data.students && data.students.length > 0) {
        let html = '<h3>✅ Recognized Students:</h3>';
        data.students.forEach(student => {
            html += `
                <div class="student-card">
                    <h3>👤 ${student.name}</h3>
                    <p><strong>Reg No:</strong> ${student.reg_no}</p>
                    <p><strong>Semester:</strong> ${student.semester}</p>
                    <p><strong>Phone:</strong> ${student.phone}</p>
                    <p><strong>Confidence:</strong> ${(student.confidence * 100).toFixed(2)}%</p>
                    <p><strong>Time:</strong> ${data.timestamp}</p>
                </div>
            `;
        });
        recognitionResult.innerHTML = html;
        recognitionResult.className = 'result-box success';
    } else {
        recognitionResult.innerHTML = '<p>❌ No faces recognized. Please try again with better lighting or position.</p>';
        recognitionResult.className = 'result-box error';
    }
    recognitionResult.style.display = 'block';
}

function showSuccess(element, message) {
    element.innerHTML = `<p>✅ ${message}</p>`;
    element.className = 'result-box success';
    element.style.display = 'block';
}

function showError(element, message) {
    element.innerHTML = `<p>❌ ${message}</p>`;
    element.className = 'result-box error';
    element.style.display = 'block';
}

// Initialize
window.addEventListener('load', () => {
    console.log('Application initialized');
    console.log('Connecting to server...');
});

// Handle page visibility (pause video when tab is hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - pausing streams');
    } else {
        console.log('Page visible');
    }
});

