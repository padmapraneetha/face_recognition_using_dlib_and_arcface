import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import tempfile
import time
import insightface
from insightface.app import FaceAnalysis

# ───────────────────────── session state ─────────────────────────
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = []
if 'known_names' not in st.session_state:
    st.session_state.known_names = []

# keep photos between reruns
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'upload_image' not in st.session_state:
    st.session_state.upload_image = None

DATASET_FILE = "face_dataset.pkl"

# Initialize ArcFace
app = FaceAnalysis()
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU

# ───────────────────────── persistence ───────────────────────────
def load_dataset():
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, 'rb') as f:
                data = pickle.load(f)
                st.session_state.known_faces = data.get('faces', [])
                st.session_state.known_names = data.get('names', [])
        except Exception:
            st.session_state.known_faces, st.session_state.known_names = [], []

def save_dataset():
    with open(DATASET_FILE, 'wb') as f:
        pickle.dump({'faces': st.session_state.known_faces,
                     'names': st.session_state.known_names}, f)

# ───────────────────────── face helpers ──────────────────────────
def register_face(image: Image.Image, name: str):
    img = np.array(image)
    faces = app.get(img)
    if not faces:
        return False, "No face detected."
    if name in st.session_state.known_names:
        return False, f'"{name}" is already registered.'
    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding
    st.session_state.known_faces.append(embedding.tolist())
    st.session_state.known_names.append(name)
    save_dataset()
    return True, "Face registered successfully!"

def recognize_faces(image: Image.Image):
    img = np.array(image)
    faces = app.get(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    names_out = []

    for face in faces:
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        top, right, bottom, left = bbox[1], bbox[2], bbox[3], bbox[0]  # Map to dlib-style (top, right, bottom, left)
        name = "Unknown"
        score = None
        valid_faces = [np.array(f) for f in st.session_state.known_faces if f is not None and isinstance(f, (list, np.ndarray))]
        if valid_faces:
            known_faces = np.stack(valid_faces)
            embedding_norm = embedding / np.linalg.norm(embedding)
            known_faces_norm = known_faces / np.linalg.norm(known_faces, axis=1, keepdims=True)
            cosine_similarities = np.dot(known_faces_norm, embedding_norm)
            idx = np.argmax(cosine_similarities)
            score = cosine_similarities[idx]
            if score > 0.35:  # ArcFace threshold
                name = st.session_state.known_names[idx]

        label = f"{name} ({score:.2f})" if score is not None else name
        names_out.append((name, score))

        cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(img_rgb, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(img_rgb, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return img_rgb, names_out

# ───────────────────────── streamlit UI ─────────────────────────
def main():
    st.title("Face Recognition System with ArcFace")
    load_dataset()

    option = st.sidebar.selectbox(
        "Choose an option:",
        ["Register Face", "Recognize from Camera",
         "Recognize from Upload", "View Dataset"]
    )

    if option == "Register Face":
        st.header("Register New Face")
        name = st.text_input("Enter name:")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Capture from Camera")
            if st.button("📷 Capture Photo"):
                st.session_state.captured_image = capture_from_camera()
                if st.session_state.captured_image:
                    st.image(st.session_state.captured_image, width=300)

            if st.session_state.captured_image is not None:
                if st.button("✅ Register This Face"):
                    ok, msg = register_face(st.session_state.captured_image, name.strip())
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.balloons()
                        st.session_state.captured_image = None

        with col2:
            st.subheader("Upload Image")
            uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded:
                st.session_state.upload_image = Image.open(uploaded)
                st.image(st.session_state.upload_image, width=300)

            if st.session_state.upload_image is not None:
                if st.button("Register Uploaded Face"):
                    ok, msg = register_face(st.session_state.upload_image, name.strip())
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.session_state.upload_image = None
                        st.balloons()

    elif option == "Recognize from Camera":
        st.header("Recognize Face from Camera")
        if st.button("📷 Capture and Recognize"):
            if not st.session_state.known_faces:
                st.warning("No faces registered.")
            else:
                img = capture_from_camera()
                if img:
                    t0 = time.time()
                    res, names = recognize_faces(img)
                    st.image(res, width=400,
                             caption=f"⏱ {(time.time()-t0)*1000:.1f} ms")
                    for name, score in names:
                        if score is not None:
                            st.write(f"**{name}** - Score: `{score:.2f}`")
                        else:
                            st.write(f"**{name}**")

    elif option == "Recognize from Upload":
        st.header("Recognize Face from Upload")
        up = st.file_uploader("Choose image or video...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        if up and st.session_state.known_faces:
            if up.type.startswith("image"):
                image = Image.open(up)
                st.image(image)
                if st.button("Recognize Faces"):
                    t0 = time.time()
                    res, names = recognize_faces(image)
                    st.image(res, caption=f"⏱ {(time.time()-t0)*1000:.1f} ms")
                    for name, score in names:
                        if score is not None:
                            st.write(f"**{name}** - Score: `{score:.2f}`")
                        else:
                            st.write(f"**{name}**")
            else:
                st.video(up)
                if st.button("Process Video"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(up.read())
                        vid_path = tmp.name
                    cap = cv2.VideoCapture(vid_path)
                    frames, hits, total = 0, set(), 0.0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames += 1
                        if frames % 30 == 0:
                            t0 = time.time()
                            res, names = recognize_faces(
                                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                            total += time.time() - t0
                            hits.update([name for name, _ in names])
                    cap.release()
                    os.unlink(vid_path)
                    st.success(f"Frames: {frames} | Total recog time: {total:.2f}s")
                    st.write("Faces:", ", ".join(hits) if hits else "None")

    elif option == "View Dataset":
        st.header("Registered Faces")
        if not st.session_state.known_names:
            st.info("No faces registered.")
        else:
            st.write(f"Total: {len(st.session_state.known_names)}")
            for i, n in enumerate(st.session_state.known_names):
                c1, c2 = st.columns([4, 1])
                c1.write(f"{i+1}. {n}")
                if c2.button("Delete", key=f"del{i}"):
                    st.session_state.known_faces.pop(i)
                    st.session_state.known_names.pop(i)
                    save_dataset()
                    st.rerun()
            if st.button("Clear All"):
                st.session_state.known_faces.clear()
                st.session_state.known_names.clear()
                save_dataset()
                st.success("Cleared.")
                st.rerun()

def capture_from_camera():
    cam = None
    try:
        for idx in (0, 1, 2):
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                break
            cam.release()

        if not cam or not cam.isOpened():
            st.error("Could not access camera.")
            return None

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(30):
            cam.read()

        ret, frame = cam.read()
        if not (ret and frame is not None):
            st.error("Failed to capture image.")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    finally:
        if cam:
            cam.release()

if __name__ == "__main__":
    main()