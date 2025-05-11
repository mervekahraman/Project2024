import ssl
# ─────────────────────────────────────────────────────────────────────────────
# macOS SSL workaround for MediaPipe downloads
ssl._create_default_https_context = ssl._create_unverified_context

import os, time
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

from transformers import T5Tokenizer, T5ForConditionalGeneration

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load instruction-tuned T5
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_t5():
    tok = T5Tokenizer.from_pretrained("google/flan-t5-small")
    mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tok, mdl

tok, t5 = load_t5()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Helpers to restrict generation
# ─────────────────────────────────────────────────────────────────────────────
def build_allowed_token_ids(glosses: list[str], tokenizer: T5Tokenizer):
    allowed = set()
    for w in glosses:
        subtoks = tokenizer.tokenize(w)
        ids     = tokenizer.convert_tokens_to_ids(subtoks)
        allowed.update(ids)
    for p in [" ", ".", ","]:
        pid = tokenizer.convert_tokens_to_ids(p)
        if pid is not None:
            allowed.add(pid)
    return allowed

def only_allowed_tokens_fn(allowed_ids: set[int]):
    def callback(batch_id: int, input_ids):
        return list(allowed_ids)
    return callback

# ─────────────────────────────────────────────────────────────────────────────
# 3) CLASS_MAP (0–225) → just the gloss words
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 4) Custom Attention Layer (if used)
# ─────────────────────────────────────────────────────────────────────────────
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=(input_shape[-1],1),
            initializer='uniform', trainable=True
        )
        super().build(input_shape)
    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Load TSL model + normalization
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tsl_model(path: str):
    if not os.path.exists(path):
        st.error(f"TSL model not found:\n  {path}")
        return None
    m = tf.keras.models.load_model(
        path,
        custom_objects={'Attention':Attention},
        compile=False
    )
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seq)
    mask = seq!=0
    if mask.any():
        μ,σ = seq[mask].mean(), seq[mask].std()
        out[mask] = (seq[mask]-μ)/(σ if σ>1e-6 else 1.0)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 6) MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# ─────────────────────────────────────────────────────────────────────────────
# 7) Main Streamlit app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="TSL→Sentence", layout="wide")
    st.title("Turkish Sign Language Interpreter + T5")

    # Sidebar settings
    st.sidebar.header("Settings")
    tsl_path    = st.sidebar.text_input(
        "TSL model path",
        value="/Users/mervekahraman/Desktop/Bitirme_projesi/tsl_simple_model_v5.keras"
    )
    frame_skip  = st.sidebar.slider("Frame Skip", 1,5,2)
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1,0.9,0.4)
    show_kp     = st.sidebar.checkbox("Show Landmarks", True)

    # Load TSL model
    tsl_model = load_tsl_model(tsl_path)
    if tsl_model is None:
        st.stop()
    st.sidebar.success("✅ TSL model loaded")

    # session_state for gloss indices
    if 'seq_idxs' not in st.session_state:
        st.session_state.seq_idxs = []

    # Refresh button
    if st.sidebar.button("🔄 Refresh Sentence"):
        st.session_state.seq_idxs = []

    # Layout
    col1, col2 = st.columns([3,1])
    with col1:
        st.header("Camera Feed")
        cam_pl = st.empty()
        start  = st.button("Start Camera")
    with col2:
        st.header("Recognition & Sentence")
        pred_pl = st.empty()
        sent_pl = st.empty()

    if start:
        buffer = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        ) as hol:
            fcount, cur_pred, cur_conf = 0, "Waiting...", 0.0
            busy = False

            while True:
                ret, frame = cap.read()
                if not ret: break
                fcount += 1
                if fcount % frame_skip != 0: continue

                # preprocess
                frame = cv2.flip(frame,1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hol.process(rgb)

                # draw
                disp = rgb.copy()
                if show_kp and res.pose_landmarks:
                    mp_drawing.draw_landmarks(disp, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if res.left_hand_landmarks:
                        mp_drawing.draw_landmarks(disp, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if res.right_hand_landmarks:
                        mp_drawing.draw_landmarks(disp, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cam_pl.image(disp, channels="RGB", use_container_width=True)

                # extract & buffer keypoints
                pose = (np.array([[lm.x,lm.y,lm.z,lm.visibility]
                                 for lm in res.pose_landmarks.landmark]).flatten()
                        if res.pose_landmarks else np.zeros(33*4))
                lh = (np.array([[lm.x,lm.y,lm.z]
                               for lm in res.left_hand_landmarks.landmark]).flatten()
                      if res.left_hand_landmarks else np.zeros(21*3))
                rh = (np.array([[lm.x,lm.y,lm.z]
                               for lm in res.right_hand_landmarks.landmark]).flatten()
                      if res.right_hand_landmarks else np.zeros(21*3))
                kp = np.concatenate([pose, lh, rh])

                if np.any(kp):
                    buffer.append(kp)
                    if len(buffer) > 30:
                        buffer.pop(0)

                # predict gloss
                if len(buffer) == 30 and not busy:
                    busy = True
                    seq  = normalize_sequence(np.array(buffer))
                    pred = tsl_model.predict(seq[np.newaxis,...], verbose=0)[0]
                    idx  = int(pred.argmax())
                    conf = float(pred[idx])
                    if conf >= conf_thresh:
                        st.session_state.seq_idxs.append(idx)
                        cur_pred, cur_conf = CLASS_MAP.get(idx, f"Class{idx}"), conf
                    buffer = buffer[-15:]
                    busy = False

                pred_pl.markdown(f"**Sign:** {cur_pred} ({cur_conf:.2f})")

                # assemble sentence
                if st.session_state.seq_idxs:
                    glosses   = [CLASS_MAP[i] for i in st.session_state.seq_idxs]
                    allowed   = build_allowed_token_ids(glosses, tok)
                    inputs    = tok(" ".join(glosses), return_tensors="pt")
                    outs      = t5.generate(
                        **inputs,
                        max_length=len(glosses)*3,
                        prefix_allowed_tokens_fn=only_allowed_tokens_fn(allowed),
                        num_beams=3,
                        early_stopping=True
                    )
                    sentence  = tok.decode(outs[0], skip_special_tokens=True)
                    sentence  = sentence.capitalize().rstrip() + "."
                    sent_pl.markdown(f"**Sentence:** {sentence}")

                time.sleep(0.01)

            cap.release()

    else:
        cam_pl.image(
            "https://via.placeholder.com/640x480.png?text=Camera+Off",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
