import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os

# Keras / TensorFlow imports
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

# ------------------------------
# 1) Custom Attention Layer
# ------------------------------
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], 1),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()

# ------------------------------
# 2) Model Loader
# ------------------------------
@st.cache_resource
def load_tsl_model(path: str):
    # 1) Check existence
    if not os.path.exists(path):
        st.error(f"Model path does not exist:\n  {path}")
        return None
    # 2) Try loading
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'Attention': Attention},
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model:\n  {e}")
        return None

# ------------------------------
# 3) Normalization Utility
# ------------------------------
def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    norm_seq = np.zeros_like(sequence)
    mask = sequence != 0
    if np.any(mask):
        mean = sequence[mask].mean()
        std  = sequence[mask].std()
        if std > 1e-6:
            norm_seq[mask] = (sequence[mask] - mean) / std
        else:
            norm_seq[mask] = sequence[mask] - mean
    return norm_seq

# ------------------------------
# 4) MediaPipe Setup
# ------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ------------------------------
# 5) Full CLASS_MAP
# ------------------------------
CLASS_MAP = {
    0: "abla (sister)", 1: "acele (hurry)", 2: "acikmak (hungry)",
    3: "afiyet_olsun (enjoy_your_meal)", 4: "agabey (brother)",
    5: "agac (tree)", 6: "agir (heavy)", 7: "aglamak (cry)",
    8: "aile (family)", 9: "akilli (wise)", 10: "akilsiz (unwise)",
    11: "akraba (kin)", 12: "alisveris (shopping)", 13: "anahtar (key)",
    14: "anne (mother)", 15: "arkadas (friend)", 16: "ataturk (ataturk)",
    17: "ayakkabi (shoe)", 18: "ayna (mirror)", 19: "ayni (same)",
    20: "baba (father)", 21: "bahce (garden)", 22: "bakmak (look)",
    23: "bal (honey)", 24: "bardak (glass)", 25: "bayrak (flag)",
    26: "bayram (feast)", 27: "bebek (baby)", 28: "bekar (single)",
    29: "beklemek (wait)", 30: "ben (I)", 31: "benzin (petrol)",
    32: "beraber (together)", 33: "bilgi_vermek (inform)", 34: "biz (we)",
    35: "calismak (work)", 36: "carsamba (wednesday)", 37: "catal (fork)",
    38: "cay (tea)", 39: "caydanlik (teapot)", 40: "cekic (hammer)",
    41: "cirkin (ugly)", 42: "cocuk (child)", 43: "corba (soup)",
    44: "cuma (friday)", 45: "cumartesi (saturday)", 46: "cuzdan (wallet)",
    47: "dakika (minute)", 48: "dede (grandfather)", 49: "degistirmek (change)",
    50: "devirmek (topple)", 51: "devlet (government)", 52: "doktor (doctor)",
    53: "dolu (full)", 54: "dugun (wedding)", 55: "dun (yesterday)",
    56: "dusman (enemy)", 57: "duvar (wall)", 58: "eczane (pharmacy)",
    59: "eldiven (glove)", 60: "emek (labor)", 61: "emekli (retired)",
    62: "erkek (male)", 63: "et (meal)", 64: "ev (house)", 65: "evet (yes)",
    66: "evli (married)", 67: "ezberlemek (memorize)", 68: "fil (elephant)",
    69: "fotograf (photograph)", 70: "futbol (football)", 71: "gecmis (past)",
    72: "gecmis_olsun (get_well)", 73: "getirmek (bring)", 74: "gol (lake)",
    75: "gomlek (shirt)", 76: "gormek (see)", 77: "gostermek (show)",
    78: "gulmek (laugh)", 79: "hafif (lightweight)", 80: "hakli (right)",
    81: "hali (carpet)", 82: "hasta (ill)", 83: "hastane (hospital)",
    84: "hata (fault)", 85: "havlu (towel)", 86: "hayir (no)",
    87: "hayirli_olsun (congratulations)", 88: "hayvan (animal)",
    89: "hediye (gift)", 90: "helal (halal)", 91: "hep (always)",
    92: "hic (never)", 93: "hoscakal (goodbye)", 94: "icmek (drink)",
    95: "igne (needle)", 96: "ilac (medicine)", 97: "ilgilenmemek (not_interested)",
    98: "isik (light)", 99: "itmek (push)", 100: "iyi (good)",
    101: "kacmak (escape)", 102: "kahvalti (breakfast)", 103: "kalem (pencil)",
    104: "kalorifer (radiator)", 105: "kapi (door)", 106: "kardes (sibling)",
    107: "kavsak (crossroads)", 108: "kaza (accident)", 109: "kemer (belt)",
    110: "keske (if_only)", 111: "kim (who)", 112: "kimlik (identity)",
    113: "kira (rent)", 114: "kitap (book)", 115: "kiyma (mince)",
    116: "kiz (female)", 117: "koku (smell)", 118: "kolonya (cologne)",
    119: "komur (coal)", 120: "kopek (dog)", 121: "kopru (bridge)",
    122: "kotu (bad)", 123: "kucak (lap)", 124: "leke (stain)",
    125: "maas (salary)", 126: "makas (scissors)", 127: "masa (tongs)",
    128: "masallah (god_preserve)", 129: "melek (angel)", 130: "memnun_olmak (be_pleased)",
    131: "mendil (napkin)", 132: "merdiven (stairs)", 133: "misafir (guest)",
    134: "mudur (manager)", 135: "musluk (tap)", 136: "nasil (how)",
    137: "neden (why)", 138: "nerede (where)", 139: "nine (grandmother)",
    140: "ocak (oven)", 141: "oda (room)", 142: "odun (wood)",
    143: "ogretmen (teacher)", 144: "okul (school)", 145: "olimpiyat (olympiad)",
    146: "olmaz (nope)", 147: "olur (allright)", 148: "onlar (they)",
    149: "orman (forest)", 150: "oruc (fasting)", 151: "ozur_dilemek (apologize)",
    152: "pamuk (cotton)", 153: "pantolon (trousers)", 154: "para (money)",
    155: "pastirma (pastrami)", 156: "patates (potato)", 157: "pazar (sunday)",
    158: "pazartesi (monday)", 159: "pencere (window)", 160: "persembe (thursday)",
    161: "piknik (picnic)", 162: "polis (police)", 163: "psikoloji (psychology)",
    164: "rica_etmek (request)", 165: "saat (hour)", 166: "sabun (soap)",
    167: "salca (sauce)", 168: "sali (tuesday)", 169: "sampiyon (champion)",
    170: "sapka (hat)", 171: "savas (war)", 172: "seker (sugar)",
    173: "selam (hi)", 174: "semsiye (umbrella)", 175: "sen (you)",
    176: "senet (bill)", 177: "serbest (free)", 178: "ses (voice)",
    179: "sevmek (love)", 180: "seytan (evil)", 181: "sinir (border)",
    182: "siz (you)", 183: "soylemek (say)", 184: "soz (promise)",
    185: "sut (milk)", 186: "tamam (okay)", 187: "tarak (comb)",
    188: "tarih (date)", 189: "tatil (holiday)", 190: "tatli (sweet)",
    191: "tavan (ceiling)", 192: "tehlike (danger)", 193: "telefon (telephone)",
    194: "terazi (scales)", 195: "terzi (tailor)", 196: "tesekkur (thanks)",
    197: "tornavida (screwdriver)", 198: "turkiye (turkey)", 199: "turuncu (orange)",
    200: "tuvalet (toilet)", 201: "un (flour)", 202: "uzak (far)",
    203: "uzgun (sad)", 204: "var (existing)", 205: "vergi (tax)",
    206: "yakin (near)", 207: "yalniz (alone)", 208: "yanlis (wrong)",
    209: "yapmak (do)", 210: "yarabandi (band-aid)", 211: "yardim (help)",
    212: "yarin (tomorrow)", 213: "yasak (forbidden)", 214: "yastik (pillow)",
    215: "yatak (bed)", 216: "yavas (slow)", 217: "yemek (eat)",
    218: "yemek_pisirmek (cook)", 219: "yildiz (star)", 220: "yok (absent)",
    221: "yol (road)", 222: "yorgun (tired)", 223: "yumurta (egg)",
    224: "zaman (time)", 225: "zor (difficult)"
}

# ------------------------------
# 6) Main App
# ------------------------------
def main():
    st.set_page_config(page_title="TSL Interpreter", layout="wide")
    st.title("Turkish Sign Language Interpreter")

    # Sidebar: model path & sanity checks
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input(
        "Model path",
        value="/Users/mervekahraman/Desktop/Bitirme_projesi/tsl_simple_model_v5.keras"
    )
    st.sidebar.write("Exists:", os.path.exists(model_path))
    st.sidebar.write("Is file:", os.path.isfile(model_path))
    st.sidebar.write("Is dir:", os.path.isdir(model_path))

    # Load the model
    model = load_tsl_model(model_path)
    if model is None:
        st.stop()
    st.sidebar.success("✅ Model loaded")

    # Runtime parameters
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4)
    show_keypoints = st.sidebar.checkbox("Show Landmarks", value=True)

    # Layout
    cam_col, pred_col = st.columns([3, 1])
    with cam_col:
        st.header("Camera")
        cam_placeholder = st.empty()
        start = st.button("Start Camera")
    with pred_col:
        st.header("Recognition")
        pred_text = st.empty()
        conf_bar = st.empty()
        st.header("Buffer")
        buf_prog = st.progress(0)
        buf_count = st.empty()
        st.subheader("Debug")
        debug_box = st.empty()

    if start:
        buffer = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        ) as holistic:
            frame_count = 0
            current_pred = "Waiting..."
            current_conf = 0.0
            processing = False

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera error")
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)

                    # Draw landmarks
                    disp = rgb.copy()
                    if show_keypoints and results.pose_landmarks:
                        mp_drawing.draw_landmarks(disp, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(disp, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(disp, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    cam_placeholder.image(disp, channels="RGB", use_container_width=True)

                    # Extract & buffer keypoints
                    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                      for lm in results.pose_landmarks.landmark]).flatten()
                            if results.pose_landmarks else np.zeros(33 * 4))
                    lh = (np.array([[lm.x, lm.y, lm.z]
                                    for lm in results.left_hand_landmarks.landmark]).flatten()
                          if results.left_hand_landmarks else np.zeros(21 * 3))
                    rh = (np.array([[lm.x, lm.y, lm.z]
                                    for lm in results.right_hand_landmarks.landmark]).flatten()
                          if results.right_hand_landmarks else np.zeros(21 * 3))
                    kp = np.concatenate([pose, lh, rh])

                    if np.any(kp):
                        buffer.append(kp)
                        if len(buffer) > 30:
                            buffer.pop(0)

                    # Update buffer UI
                    ratio = len(buffer) / 30
                    buf_prog.progress(ratio)
                    buf_count.text(f"{len(buffer)}/30 frames")

                    # Predict when full
                    if len(buffer) == 30 and not processing:
                        processing = True
                        seq = normalize_sequence(np.array(buffer))
                        pred = model.predict(seq[np.newaxis, ...], verbose=0)[0]
                        idx = int(pred.argmax())
                        conf = float(pred[idx])

                        # Show top‑5 debug
                        top5 = np.argsort(pred)[-5:][::-1]
                        debug_box.text("\n".join(f"{CLASS_MAP[i]}: {pred[i]:.3f}" for i in top5))

                        if conf >= confidence_threshold:
                            current_pred = CLASS_MAP.get(idx, f"Class {idx}")
                            current_conf = conf

                        buffer = buffer[-15:]
                        processing = False

                    # Display result
                    style = "color:green;" if current_conf >= confidence_threshold else "color:gray;"
                    pred_text.markdown(f"<h2 style='text-align:center;{style}'>{current_pred}</h2>",
                                       unsafe_allow_html=True)
                    conf_bar.progress(current_conf)

                    time.sleep(0.01)

            except Exception as e:
                st.error(f"Runtime error:\n  {e}")
            finally:
                cap.release()

    else:
        cam_placeholder.image(
            "https://via.placeholder.com/640x480.png?text=Camera+Off",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
