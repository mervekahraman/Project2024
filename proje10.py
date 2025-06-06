import os
import ssl
import time
import numpy as np
import streamlit as st

# Bypass MacOS SSL if needed
ssl._create_default_https_context = ssl._create_unverified_context

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

from transformers import T5Tokenizer, T5ForConditionalGeneration
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load your T5 sentence generator
# ─────────────────────────────────────────────────────────────────────────────
tok = T5Tokenizer.from_pretrained("Turkish-NLP/t5-efficient-small-turkish")
t5  = T5ForConditionalGeneration.from_pretrained("Turkish-NLP/t5-efficient-small-turkish")

def t5_generate(glosses):
    prompt = "gloss: " + " ".join(glosses)
    inputs = tok(prompt, return_tensors="pt").to(t5.device)
    outputs = t5.generate(**inputs, max_length=32)
    return tok.decode(outputs[0], skip_special_tokens=True).capitalize() + "."

# ─────────────────────────────────────────────────────────────────────────────
# 2) Custom Attention (if your model needs it)
# ─────────────────────────────────────────────────────────────────────────────
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kw): 
        self.supports_masking = True
        super().__init__(**kw)
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=(input_shape[-1],1),
                                      initializer="uniform",
                                      trainable=True)
        super().build(input_shape)
    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Load your TSL recognition model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_tsl_model(path):
    if not os.path.exists(path):
        st.error(f"TSL model not found: {path}")
        return None
    m = tf.keras.models.load_model(path,
        custom_objects={'Attention': Attention}, compile=False)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

# ─────────────────────────────────────────────────────────────────────────────
# 4) Load bundled MediaPipe TFLite models
# ─────────────────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
HOLISTIC = mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Helpers: normalize + extract keypoints
# ─────────────────────────────────────────────────────────────────────────────
def normalize_seq(seq: np.ndarray):
    out = np.zeros_like(seq)
    mask = seq != 0
    if mask.any():
        μ,σ = seq[mask].mean(), seq[mask].std()
        out[mask] = (seq[mask]-μ)/σ if σ>1e-6 else (seq[mask]-μ)
    return out

def extract_kp(results):
    pose = (np.array([[lm.x,lm.y,lm.z,lm.visibility] 
             for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33*4))
    lh   = (np.array([[lm.x,lm.y,lm.z] 
             for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21*3))
    rh   = (np.array([[lm.x,lm.y,lm.z] 
             for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21*3))
    return np.concatenate([pose, lh, rh])

# ─────────────────────────────────────────────────────────────────────────────
# 6) Your class mapping
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
# 7) Video transformer: runs in the server per‐frame
# ─────────────────────────────────────────────────────────────────────────────
class TSLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_tsl_model("tsl_simple_model_v5.keras")
        self.buffer = []
        self.seq = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # run mediapipe
        mp_res = HOLISTIC.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        kp = extract_kp(mp_res)
        if kp.any():
            self.buffer.append(kp)
            if len(self.buffer)>30: self.buffer.pop(0)
        # predict once buffer full
        if len(self.buffer)==30:
            seq = normalize_seq(np.array(self.buffer))
            pred = self.model.predict(seq[np.newaxis,...],verbose=0)[0]
            idx  = int(pred.argmax())
            if pred[idx]>0.4:
                self.seq.append(idx)
            self.buffer = self.buffer[-15:]
        # overlay info
        cv2.putText(img, f"sig: {CLASS_MAP.get(self.seq[-1],'...')}" if self.seq else "Waiting",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return img

# ─────────────────────────────────────────────────────────────────────────────
# 8) Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("TSL Interpreter → Sentence (Cloud-friendly)")

webrtc_ctx = webrtc_streamer(
    key="tsl",
    video_transformer_factory=TSLTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_transformer:
    st.markdown("### Your Sentence so far:")
    glosses = [CLASS_MAP[i] for i in webrtc_ctx.video_transformer.seq]
    if glosses:
        st.write("• Glosses:", glosses)
        st.write("• Sentence:", t5_generate(glosses))
    else:
        st.write("No signs detected yet.")

