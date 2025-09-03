import cv2
import torchreid
from ultralytics import YOLO
import numpy as np
from numpy.linalg import norm

# ----------------------------
# Load YOLO (for person detection)
# ----------------------------
yolo_model = YOLO("yolov8n.pt")  # lightweight + fast

# ----------------------------
# 2. Load Torchreid (for re-identification)
# ----------------------------
extractor = torchreid.utils.FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',
    device='cpu'  # change to "cuda" if you have GPU
)

# ----------------------------
# 3. Single-camera setup (Camera 2)
# ----------------------------
cap = cv2.VideoCapture(2)  # camera 2 only
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------------------
# 4. Global ReID database
# ----------------------------
global_id = 0
person_db = {}  # {id: [feature_vectors]}
MAX_FEATS = 20  # keep last 20 embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def match_id(features, db, threshold=0.75):
    """Match new features to existing DB using cosine similarity"""
    best_match, best_score = None, 0
    for pid, feats in db.items():
        ref_feat = np.mean(feats, axis=0)
        score = cosine_similarity(features, ref_feat)
        if score > best_score:
            best_score, best_match = score, pid
    if best_score > threshold:
        return best_match
    return None

# ----------------------------
# 5. Main loop
# ----------------------------
while True:
    if not cap.isOpened():
        print("❌ Camera 2 not available")
        break

    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO detection
    results = yolo_model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue  # only detect persons

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Convert BGR -> RGB for ReID
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            features = extractor([person_crop])[0].detach().cpu().numpy()

            # Match with database
            pid = match_id(features, person_db)

            if pid is None:
                global_id += 1
                person_db[global_id] = [features]
                pid = global_id
            else:
                person_db[pid].append(features)
                if len(person_db[pid]) > MAX_FEATS:
                    person_db[pid] = person_db[pid][-MAX_FEATS:]

            # Draw bounding box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 6. Release resources
# ----------------------------
cap.release()
cv2.destroyAllWindows()
