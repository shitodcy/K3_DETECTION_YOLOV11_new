import streamlit as st
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import time

# ==========================================
# 1. KONFIGURASI HALAMAN & UI BERSIH
# ==========================================
st.set_page_config(page_title="Sistem Deteksi K3", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Memberikan sedikit padding pada metric box agar terlihat seperti kartu */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #303040;
        padding: 15px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INISIALISASI MODEL
# ==========================================
MODEL_PATH = "/home/azunya/Documents/kuliah/sem6/matkul/data mining/project/K3_DETECTION_YOLOV11-new/runs/detect/K3_DETECTION/yolov11n_roboflow100/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
CLASSES_PELANGGARAN = ['no-helmet', 'no-vest']

if 'cooldown_tracker' not in st.session_state:
    st.session_state.cooldown_tracker = {}

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("Sistem K3 YOLOv11")
    st.markdown("---")
    
    app_mode = st.radio("Pilih Mode", ["Live Stream", "Analisa Gambar"])
    st.markdown("---")
    
    st.subheader("Konfigurasi Model")
    confidence_threshold = st.slider("Minimal Akurasi", 0.0, 1.0, 0.40, 0.05)
    
    st.subheader("Penyimpanan Bukti")
    output_dir = st.text_input("Folder Output", value="bukti_pelanggaran_k3")
    cooldown_time = st.number_input("Jeda Capture (detik)", min_value=1.0, value=10.0)

# ==========================================
# 4. KONTEN UTAMA
# ==========================================
if app_mode == "Live Stream":
    st.header("Live Stream")
    
    run_camera = st.toggle("Aktifkan Kamera", value=False)
    st.markdown("---")
    
    FRAME_WINDOW = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Kamera tidak dapat diakses.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            prev_time = time.time()
            
            while run_camera:
                success, frame = cap.read()
                if not success: break
                    
                frame = cv2.flip(frame, 1)
                
                results = model.track(frame, conf=confidence_threshold, persist=True, verbose=False)
                annotated_frame = results[0].plot()
                
                # Kalkulasi FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Auto-Save
                if results[0].boxes.id is not None:
                    tracker_ids = results[0].boxes.id.int().cpu().tolist()
                    class_indices = results[0].boxes.cls.cpu().tolist()
                    
                    for track_id, cls_idx in zip(tracker_ids, class_indices):
                        class_name = model.names[int(cls_idx)]

                        if class_name in CLASSES_PELANGGARAN:
                            current_time = time.time()
                            last_capture = st.session_state.cooldown_tracker.get(track_id, 0)

                            if (current_time - last_capture) > cooldown_time:
                                cat_folder = os.path.join(output_dir, class_name)
                                os.makedirs(cat_folder, exist_ok=True)
                                
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"{class_name}_ID-{track_id}_{timestamp}.jpg"
                                filepath = os.path.join(cat_folder, filename)
                                cv2.imwrite(filepath, annotated_frame)
                                
                                st.session_state.cooldown_tracker[track_id] = current_time
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, use_container_width=True)
                
            cap.release()
    else:
        st.info("Kamera dinonaktifkan.")

elif app_mode == "Analisa Gambar":
    st.header("Analisa Gambar Statis")
    
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        run_detection = st.button("Jalankan Deteksi K3", type="primary", use_container_width=True)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if run_detection:
            with col2:
                with st.spinner("Menganalisa gambar..."):
                    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    results = model.predict(source=img_bgr, conf=confidence_threshold)
                    res_plotted = results[0].plot()
                    
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, caption="Hasil Deteksi", use_container_width=True)
            
            # ==========================================
            # BAGIAN RINCIAN UI BARU YANG LEBIH RAPI
            # ==========================================
            st.markdown("---")
            st.subheader("Rincian Hasil Deteksi")
            
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                st.info("Tidak ada objek yang terdeteksi pada gambar ini.")
            else:
                data_tabel = []
                jumlah_pekerja = 0
                jumlah_apd = 0
                jumlah_pelanggaran = 0
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Logika perhitungan yang lebih akurat berdasarkan kelas
                    if class_name == 'person':
                        jumlah_pekerja += 1
                        status = "Terdeteksi"
                    elif class_name in ['helmet', 'vest']:
                        jumlah_apd += 1
                        status = "Aman (APD Dipakai)"
                    elif class_name in CLASSES_PELANGGARAN:
                        jumlah_pelanggaran += 1
                        status = "Pelanggaran"
                    else:
                        status = "Terdeteksi"
                        
                    data_tabel.append({
                        "ID": i + 1,
                        "Objek": class_name.upper(),
                        "Akurasi": conf, # Biarkan sebagai float untuk progress bar
                        "Status Keselamatan": status
                    })
                
                # 1. Menampilkan Summary Card (Kartu Metrik) di atas tabel
                met_col1, met_col2, met_col3 = st.columns(3)
                met_col1.metric("Pekerja Terdeteksi", jumlah_pekerja)
                met_col2.metric("Kelengkapan APD (Aman)", jumlah_apd)
                met_col3.metric("Indikasi Pelanggaran", jumlah_pelanggaran)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
               
                
                # 3. Menampilkan Tabel Interaktif dengan Column Configuration
                df = pd.DataFrame(data_tabel)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ID": st.column_config.NumberColumn(
                            "No.",
                            width="small"
                        ),
                        "Objek": st.column_config.TextColumn(
                            "Kategori Objek",
                            width="medium"
                        ),
                        "Akurasi": st.column_config.ProgressColumn(
                            "Akurasi Model",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Status Keselamatan": st.column_config.TextColumn(
                            "Status Verifikasi",
                            width="large"
                        )
                    }
                )

        else:
            with col2:
                st.info("Klik tombol 'Jalankan Deteksi K3' di atas untuk melihat hasil visual.")
