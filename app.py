import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import os
import pandas as pd
from datetime import datetime
import time
from fpdf import FPDF

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="SiberKalkan Yönetim Paneli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TASARIMI (PROFESYONEL VERSİYON) ---
st.markdown("""
<style>
    /* Genel Buton Stili */
    div.stButton > button:first-child {
        background-color: #20B2AA;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #178a84;
        transform: scale(1.02);
    }

    /* --- GİRİŞ EKRANI (LOGIN CARD) --- */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px;
    }
    .login-card {
        background-color: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        width: 100%;
        max-width: 500px;
        border-top: 6px solid #20B2AA;
    }
    .login-logo { font-size: 60px; margin-bottom: 15px; }
    .login-title { font-size: 26px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
    .login-subtitle { font-size: 16px; color: #7f8c8d; margin-bottom: 30px; }

    /* --- TABLET TASARIMI --- */
    .tablet-frame {
        max_width: 700px;
        margin: auto;
        border: 16px solid #34495e;
        border-radius: 36px;
        background-color: #E5DDD5;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
    }
    
    .tablet-header-bar {
        background-color: #075E54;
        color: white;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Helvetica Neue', sans-serif;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        z-index: 10;
    }
    
    .header-user { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 16px; }
    .header-title { font-size: 14px; opacity: 0.8; }
    .header-score { 
        background-color: rgba(255,255,255,0.2); 
        padding: 5px 12px; 
        border-radius: 20px; 
        font-weight: bold; 
        font-size: 14px;
        color: #FFD700;
    }

    .chat-area {
        height: 450px;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    /* Mesaj Baloncukları */
    .msg-incoming {
        align-self: flex-start;
        background-color: white;
        color: #333;
        padding: 10px 16px;
        border-radius: 0 16px 16px 16px;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.4;
    }
    .msg-outgoing {
        align-self: flex-end;
        background-color: #dcf8c6;
        color: #333;
        padding: 10px 16px;
        border-radius: 16px 0 16px 16px;
        max-width: 80%;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.4;
    }
    .msg-ghost {
        align-self: flex-end;
        background-color: rgba(255, 200, 200, 0.5);
        color: #c0392b;
        padding: 10px 16px;
        border-radius: 16px 0 16px 16px;
        max-width: 80%;
        border: 2px dashed #e74c3c;
        text-align: left;
        font-size: 14px;
    }

    /* Sakinleşme Animasyonları */
    @keyframes pulse-ring {
        0% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 30px rgba(33, 150, 243, 0); }
        100% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
    }
    .calm-circle {
        width: 120px; height: 120px; background: linear-gradient(135deg, #2196F3, #21CBF3);
        color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 50px; font-weight: bold; margin: 20px auto;
        box-shadow: 0 10px 20px rgba(33, 150, 243, 0.4);
        animation: pulse-ring 2s infinite;
    }
    .calm-container {
        text-align: center; padding: 40px; background: white; height: 100%;
        display: flex; flex-direction: column; justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'backend'
if 'user_score' not in st.session_state: st.session_state.user_score = 100
if 'history' not in st.session_state: st.session_state.history = []
if 'chat_log' not in st.session_state: st.session_state.chat_log = [{"role": "incoming", "text": "Selam! Naber?"}]
if 'train_key_counter' not in st.session_state: st.session_state.train_key_counter = 0
if 'sim_mode' not in st.session_state: st.session_state.sim_mode = "Oyun Modu (Puanlı)"
if 'breathing_phase' not in st.session_state: st.session_state.breathing_phase = False 
if 'student_name' not in st.session_state: st.session_state.student_name = ""
if 'chat_turn' not in st.session_state: st.session_state.chat_turn = "student" # 'student' veya 'counterpart'

# --- 4. MODEL VE FONKSİYONLAR ---
KARA_LISTE = ["siktir", "sik", "amk", "aq", "oç", "piç", "yavşak", "gerizekalı", "salak", "aptal", "mal", "defol", "şerefsiz"]
DOSYA_ADI = "veri_havuzu.xlsx"

GERI_DONUTLER = {
    "Küfür / Hakaret": "Bu dilde konuşmak saygı sınırlarını aşıyor. Lütfen daha nazik olur musun?",
    "Siber Zorbalık": "Bu yazdığın karşı tarafı üzebilir. Dijital dünyada iz bırakırken dikkatli olmalısın.",
    "Tehdit": "Tehdit içeren ifadeler yasal suç teşkil edebilir. Lütfen öfkeni kontrol et.",
    "Taciz": "Kişisel sınırlara saygı duymak önemlidir. Lütfen ifadelerini yumuşat.",
    "Genel": "Bu mesaj topluluk kurallarına aykırı görünüyor."
}

# --- PDF FONKSİYONLARI ---
def tr_pdf(text):
    degisim = str.maketrans("ğĞıİşŞçÇöÖüÜ", "gGiIsScCoOuU")
    return text.translate(degisim)

def create_pdf_report(score, history, student_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(7, 94, 84)
    pdf.cell(0, 10, tr_pdf("SiberKalkan Veli Raporu"), ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    tarih = datetime.now().strftime("%d.%m.%Y")
    pdf.cell(0, 10, tr_pdf(f"Öğrenci: {student_name} | Tarih: {tarih}"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, tr_pdf("1. DIJITAL PUAN DURUMU"), ln=True)
    
    if score >= 80: pdf.set_fill_color(76, 175, 80)
    elif score >= 50: pdf.set_fill_color(255, 152, 0)
    else: pdf.set_fill_color(244, 67, 54)
    
    bar_width = (score / 150) * 190
    if bar_width > 190: bar_width = 190
    if bar_width < 0: bar_width = 0
    pdf.rect(10, 55, bar_width, 15, 'F')
    pdf.ln(20)
    pdf.cell(0, 5, tr_pdf(f"Puan: {score}"), ln=True)
    pdf.ln(10)
    
    pdf.cell(0, 10, tr_pdf("2. TAVSIYE"), ln=True)
    pdf.set_font("Arial", '', 11)
    if score >= 80: tavsiye = f"Sayin Veli, {student_name} dijital iletisimde son derece saygili."
    elif score >= 50: tavsiye = f"Sayin Veli, {student_name} bazen duygusal tepkiler verse de kurallara uyuyor."
    else: tavsiye = f"Sayin Veli, {student_name} dijital iletisimde zorbalik egilimleri gosteriyor."
    pdf.multi_cell(0, 8, tr_pdf(tavsiye))
    
    return pdf.output(dest='S').encode('latin-1')

def kara_liste_kontrolu(metin):
    for kelime in KARA_LISTE:
        if kelime in metin.lower(): return True, kelime
    return False, None

def excel_hafiza_kontrolu(metin):
    if os.path.exists(DOSYA_ADI):
        try:
            df = pd.read_excel(DOSYA_ADI)
            bulunan = df[df['Metin'].astype(str).str.lower().str.strip() == metin.lower().strip()]
            if not bulunan.empty:
                return True, bulunan.iloc[-1]['Etiket']
        except: pass
    return False, None

@st.cache_resource
def model_yukle():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "siber_kalkan_modeli")
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")
        return tokenizer, model
    except: return None, None

tokenizer, model = model_yukle()

def veriyi_excele_kaydet(metin, etiket, skor, kaynak):
    yeni_veri = {"Tarih": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "Metin": [metin], "Etiket": [etiket], "AI_Skoru": [skor], "Kaynak": [kaynak]}
    df_yeni = pd.DataFrame(yeni_veri)
    try:
        if os.path.exists(DOSYA_ADI):
            df_eski = pd.read_excel(DOSYA_ADI)
            pd.concat([df_eski, df_yeni], ignore_index=True).to_excel(DOSYA_ADI, index=False)
        else: df_yeni.to_excel(DOSYA_ADI, index=False)
    except: pass

# ==========================================
# 💾 SAYFA 3: VERİ DÜZENLEME (HATA DÜZELTİLDİ)
# ==========================================
def show_data_editor():
    st.title("📝 Veri Seti Düzenleme")
    
    if not os.path.exists(DOSYA_ADI):
        st.error("Veri dosyası yok.")
        if st.button("GERİ"): 
            st.session_state.page='backend' 
            st.rerun()
        return

    df = pd.read_excel(DOSYA_ADI)
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True)
    
    c1, c2 = st.columns([1,4])
    with c1:
        if st.button("KAYDET", type="primary"):
            edited_df.to_excel(DOSYA_ADI, index=False)
            st.success("Kaydedildi!")
            time.sleep(0.5)
            st.rerun()
    with c2:
        if st.button("GERİ DÖN"):
            st.session_state.page = 'backend'
            st.rerun()

# ==========================================
# 🖥️ SAYFA 1: BACKEND
# ==========================================
def show_backend():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9471/9471031.png", width=100)
        st.title("SiberKalkan v1.3")
        st.caption("Yönetici Paneli")
        st.markdown("---")
        st.session_state.sim_mode = st.radio("Mod Seç:", ("Oyun Modu (Puanlı)", "Eğitim Modu (Katı Kurallı)"))
        if st.button("📲 MOBİL SİMÜLASYON", type="primary"): st.session_state.page = 'mobile'; st.rerun()
        st.markdown("---")
        if st.button("✏️ VERİLERİ DÜZENLE"): st.session_state.page = 'data_editor'; st.rerun()

    st.header("🛡️ SiberKalkan Tehdit Analiz Merkezi")
    
    # Model Eğitme Alanı
    with st.expander("🧠 Yapay Zekayı Eğit (Manuel Veri Girişi)"):
        c_e1, c_e2 = st.columns([3,1])
        with c_e1: e_txt = st.text_input("Öğretilecek Cümle:", placeholder="Örn: Yeni bir argo kelime...")
        with c_e2: e_lbl = st.selectbox("Etiket", ["Siber Zorbalık", "Normal", "Küfür/Hakaret"])
        if st.button("EĞİT VE KAYDET"):
            if e_txt:
                veriyi_excele_kaydet(e_txt, e_lbl, "1.0", "Manuel")
                st.success("Öğretildi!"); st.rerun()

    st.markdown("---")
    user_input = st.text_area("Analiz edilecek mesaj:", height=100, placeholder="Şüpheli metni buraya yapıştırın...")
    if st.button("ANALİZ ET", use_container_width=True):
        if user_input:
            kural, kelime = kara_liste_kontrolu(user_input)
            hafiza, etiket = excel_hafiza_kontrolu(user_input)
            sonuc = "Normal"
            if kural: sonuc = "Küfür / Hakaret"
            elif hafiza: sonuc = etiket
            else:
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
                outputs = model(**inputs)
                if F.softmax(outputs.logits, dim=1)[0][0].item() > 0.60: sonuc = "Siber Zorbalık"
            
            if sonuc != "Normal": st.error(f"🚨 TESPİT: {sonuc}"); veriyi_excele_kaydet(user_input, sonuc, "1.0", "AI")
            else: st.success("✅ GÜVENLİ"); veriyi_excele_kaydet(user_input, "Normal", "0.0", "AI")

# ==========================================
# 📱 SAYFA 2: TABLET SİMÜLASYONU (FINAL)
# ==========================================
def show_mobile():
    # --- ÜST BAR ---
    c_back, c_pdf = st.columns([1, 5])
    with c_back:
        if st.button("⬅️ Çıkış"): st.session_state.page = 'backend'; st.rerun()
    with c_pdf:
        if st.session_state.history and st.session_state.student_name:
            pdf_data = create_pdf_report(st.session_state.user_score, st.session_state.history, st.session_state.student_name)
            st.download_button("📄 Veli Karnesini İndir", data=pdf_data, file_name="karnem.pdf", mime="application/pdf")

    # --- 1. GİRİŞ EKRANI (CARD DESIGN) ---
    if not st.session_state.student_name:
        st.markdown("""
        <div class="login-container">
            <div class="login-card">
                <div class="login-logo">🛡️</div>
                <div class="login-title">SiberKalkan'a Hoş Geldin</div>
                <div class="login-subtitle">Güvenli bir dijital deneyim için simülasyonu başlat.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        _, c_form, _ = st.columns([1, 2, 1])
        with c_form:
            with st.form("login_form"):
                name_input = st.text_input("Adın Soyadın:", placeholder="Örn: Ali Veli")
                submitted = st.form_submit_button("SİMÜLASYONU BAŞLAT ▶", use_container_width=True)
                if submitted and name_input:
                    st.session_state.student_name = name_input
                    st.rerun()
        return

    # --- 2. SAKİNLEŞME MODU (ANIMASYON) ---
    if st.session_state.get('breathing_phase'):
        placeholder = st.empty()
        for i in range(5, 0, -1):
            placeholder.markdown(f"""
            <div class="tablet-frame" style="height: 550px; background: white; border: none;">
                <div class="calm-container">
                    <div class="calm-circle">{i}</div>
                    <h2 style="color:#0277BD;">✋ Biraz Bekleyelim...</h2>
                    <p style="color:#555; font-size:18px;">Çok öfkeli görünüyorsun. Derin bir nefes al 🧘</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.2)
        
        st.session_state.breathing_phase = False
        st.session_state.alert_active = True
        st.rerun()
        return

    # --- 3. TABLET EKRANI ---
    mode = st.session_state.sim_mode
    is_game = (mode == "Oyun Modu (Puanlı)")
    score_html = f'<div class="header-score">⭐ {st.session_state.user_score}</div>' if is_game else ''

    chat_content = ""
    for msg in st.session_state.chat_log:
        role = "msg-incoming" if msg['role'] == "incoming" else "msg-outgoing"
        chat_content += f"<div class='{role}'>{msg['text']}</div>"
    
    if st.session_state.get('alert_active'):
        chat_content += f"<div class='msg-ghost'>{st.session_state.temp_bad_msg} (İletilmedi)</div>"

    st.markdown(f"""
    <div class="tablet-frame">
        <div class="tablet-header-bar">
            <div class="header-user">👤 {st.session_state.student_name}</div>
            <div class="header-title">SiberKalkan Chat</div>
            {score_html}
        </div>
        <div class="chat-area">
            {chat_content}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 4. KONTROL / INPUT ALANI ---
    if st.session_state.get('alert_active'):
        st.error(f"⚠️ TESPİT EDİLDİ: {st.session_state.temp_reason}")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("😇 VAZGEÇ (+50 Puan)", use_container_width=True):
                st.session_state.user_score += 50
                st.session_state.alert_active = False
                st.balloons()
                veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Engellendi", "1.0", "Mobil-Vazgeçti")
                time.sleep(1); st.rerun()
        with c2:
            if is_game:
                if st.button("😈 GÖNDER (-20 Puan)", use_container_width=True):
                    st.session_state.user_score -= 20
                    st.session_state.chat_log.append({"role": "outgoing", "text": st.session_state.temp_bad_msg})
                    st.session_state.alert_active = False
                    st.session_state.chat_turn = "counterpart" 
                    veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Zorbalık", "1.0", "Mobil-İnat")
                    st.rerun()
            else:
                if st.button("✍️ MESAJI DÜZELT", use_container_width=True):
                    st.session_state.alert_active = False; st.rerun()

    else:
        # SIRA ÖĞRENCİDE
        if st.session_state.chat_turn == "student":
            with st.form("msg_form", clear_on_submit=True):
                c_txt, c_btn = st.columns([4, 1])
                with c_txt: u_msg = st.text_input("Mesajın:", placeholder="Bir şeyler yaz...", label_visibility="collapsed")
                with c_btn: sub = st.form_submit_button("GÖNDER", use_container_width=True)
                
                if sub and u_msg:
                    kural, k_kelime = kara_liste_kontrolu(u_msg)
                    hafiza, h_etiket = excel_hafiza_kontrolu(u_msg)
                    bad = False; reason = ""
                    
                    if kural: bad=True; reason=f"Yasaklı: {k_kelime}"; typ="Küfür"
                    elif hafiza: bad=True; reason=f"Hafıza: {h_etiket}"; typ=h_etiket
                    else:
                        inp = tokenizer(u_msg, return_tensors="pt", truncation=True, padding=True, max_length=64)
                        out = model(**inp)
                        if F.softmax(out.logits, dim=1)[0][0].item() > 0.60: bad=True; reason="Saldırgan Dil"; typ="Zorbalık"
                    
                    if bad:
                        st.session_state.temp_bad_msg = u_msg
                        st.session_state.temp_reason = reason
                        st.session_state.temp_type = typ
                        st.session_state.breathing_phase = True 
                        st.rerun()
                    else:
                        st.session_state.chat_log.append({"role": "outgoing", "text": u_msg})
                        if is_game: st.session_state.user_score += 10
                        st.session_state.chat_turn = "counterpart"
                        veriyi_excele_kaydet(u_msg, "Normal", "0.0", "Mobil")
                        st.rerun()

        # SIRA KARŞI TARAFTA
        else:
            st.info("💬 Karşı taraf yazıyor... (Senaryoyu devam ettir)")
            with st.form("cp_form", clear_on_submit=True):
                c_txt, c_btn = st.columns([4, 1])
                with c_txt: cp_msg = st.text_input("Karşı Tarafın Cevabı:", placeholder="Örn: Neden böyle dedin?", label_visibility="collapsed")
                with c_btn: sub = st.form_submit_button("CEVAPLA", use_container_width=True)
                
                if sub and cp_msg:
                    st.session_state.chat_log.append({"role": "incoming", "text": cp_msg})
                    st.session_state.chat_turn = "student"
                    st.rerun()

# --- YÖNLENDİRME ---
if st.session_state.page == 'backend': show_backend()
elif st.session_state.page == 'mobile': show_mobile()
elif st.session_state.page == 'data_editor': show_data_editor()
