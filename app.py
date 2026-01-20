import os
import secrets
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from model_handler import DeepFakeDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging Setup
LOG_FOLDER = 'logs'
os.makedirs(LOG_FOLDER, exist_ok=True)
import logging
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, 'activity.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize DeepFake Detector (Global for now)
# Expects weights file in root as per plan
MODEL_WEIGHTS = 'weights\dsmpe_net_epoch_4.pth' # User must provide this
detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            detector = DeepFakeDetector(weights_path=MODEL_WEIGHTS)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            detector = None
    return detector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('join')
def on_join(data):
    username = data['username']
    print(f"============= {username} has joined the channel =============")
    emit('status_message', {'msg': f'{username} has joined the channel.'}, broadcast=True)

@socketio.on('upload_media')
def handle_media_upload(data):
    # This is a bit tricky with SocketIO + File Upload. 
    # Usually easier to upload via HTTP POST and then emit event.
    # But for simplicity in this demo, let's assume client sends file via POST separately,
    # then emits 'media_shared' event with filename.
    pass 

# API for handling file upload from client
@app.route('/api/upload', methods=['POST'])
def upload_file_api():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    file = request.files['file']
    username = request.form.get('username', 'Anonymous')
    socket_id = request.form.get('socket_id')
    
    if file.filename == '':
        return {'error': 'No selected file'}, 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Deepfake Detection Logic
        det = get_detector()
        is_fake = False
        score = 0.0
        
        if det:
            try:
                # Progress Callback
                def progress_report(msg):
                    if socket_id:
                        socketio.emit('status_update', {'msg': msg}, to=socket_id)
                
                # Returns (is_fake, score, patch_scores, metadata)
                is_fake, score, patch_scores, meta = det.check_media(filepath, progress_callback=progress_report)
            except Exception as e:
                print(f"Detection failed: {e}")
                return {'status': 'error', 'message': f'Detection failed: {str(e)}'}, 500
        else:
            # Fallback if no detector
            is_fake, score, patch_scores, meta = (False, 0.0, [], {'type': 'unknown'})

        # Format patch scores for logging
        patch_log_str = ""
        if patch_scores and len(patch_scores) == 9:
            rows = [patch_scores[i:i+3] for i in range(0, 9, 3)]
            patch_log_str = "\n   Patch Analysis:\n" + "\n".join([f"   Row {i+1}: {' | '.join([f'{p:.2f}' for p in row])}" for i, row in enumerate(rows)])
        
        # Prepare Log Message
        verdict_str = "FAKE" if is_fake else "REAL"
        meta_str = ""
        if 'frames_processed' in meta:
            meta_str = f" [Video Stats: {meta['frames_processed']} processed / {meta['frames_sampled']} sampled]"
        
        log_msg = f"VERDICT: {verdict_str} | User: {username} | File: {filename} | Score: {score:.4f}{meta_str}{patch_log_str}"

        if is_fake:
            print(f"BLOCKED: {log_msg}")
            logging.warning(f"BLOCKED DEEPFAKE: {log_msg}")
            # Notify sender and Broadcast Alert
            socketio.emit('blocked_alert', {
                'username': username,
                'filename': filename,
                'score': score
            })
            
            return {
                'status': 'blocked',
                'reason': 'Deepfake Detected',
                'score': score,
                'patch_scores': patch_scores
            }, 200
        else:
            print(f"BROADCAST: {log_msg}")
            logging.info(f"BROADCASTING: {log_msg}")
            
            # Broadcast to all connected clients via SocketIO
            socketio.emit('new_media', {
                'username': username,
                'filename': filename,
                'file_url': f'/uploads/{filename}',
                'is_image': filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')),
                'score': score
            })
            return {
                'status': 'success', 
                'message': 'Media broadcasted',
                'patch_scores': patch_scores
            }, 200

if __name__ == '__main__':
    # Initialize detector on start if possible
    get_detector()
    socketio.run(app, debug=True, port=5000)
