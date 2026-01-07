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
    page_title="SiberKalkan v2.2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TASARIMI ---
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #20B2AA;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        width: 100%;
    }
    .tablet-screen-top {
        max_width: 700px;
        margin: auto;
        border: 20px solid #1f1f1f;
        border-bottom: none; 
        border-top-left-radius: 35px;
        border-top-right-radius: 35px;
        background-color: #E5DDD5;
        height: 550px; 
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-bottom: -1rem; 
    }
    [data-testid="stForm"] {
        max_width: 700px;
        margin: auto;
        border: 20px solid #1f1f1f;
        border-top: none; 
        border-bottom-left-radius: 35px;
        border-bottom-right-radius: 35px;
        background-color: #E5DDD5;
        padding: 20px;
        padding-top: 0px; 
    }
    .tablet-header {
        text-align: center;
        background-color: #075E54;
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 15px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .msg-incoming {
        align-self: flex-start;
        background-color: white;
        color: black;
        padding: 10px 14px;
        border-radius: 0 12px 12px 12px;
        max-width: 75%;
        margin-bottom: 8px;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    .msg-outgoing {
        align-self: flex-end;
        background-color: #DCF8C6;
        color: black;
        padding: 10px 14px;
        border-radius: 12px 0 12px 12px;
        max-width: 75%;
        text-align: left;
        margin-bottom: 8px;
        float: right;
        clear: both;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    }
    .msg-pending {
        background-color: rgba(255, 235, 235, 0.9);
        color: #d32f2f;
        padding: 10px 14px;
        border-radius: 12px 0 12px 12px;
        border: 2px dashed #ff5252;
        max-width: 75%;
        text-align: left;
        margin-bottom: 8px;
        float: right;
        clear: both;
    }
    .tablet-alert-box {
        background-color: #ffebee;
        color: #c62828;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ffcdd2;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 15px;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border-radius: 20px;
    }
    .guide-message {
        font-size: 14px;
        color: #555;
        font-style: italic;
        margin-top: 5px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* --- SAKİNLEŞME MODU ANİMASYONLARI --- */
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 20px rgba(33, 150, 243, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
    }
    .calm-circle {
        width: 120px; height: 120px; background-color: #039BE5; color: white;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 60px; font-weight: bold; margin: 20px;
        animation: pulse 1.5s infinite;
    }
    .calm-text {
        font-size: 24px; color: #01579B; font-weight: bold; margin-bottom: 10px;
    }
    .calm-subtext {
        font-size: 18px; color: #0277BD;
    }
    
    /* --- GİRİŞ EKRANI & YAZIYOR EFEKTİ --- */
    .login-container {
        display: flex; justify-content: center; align-items: center; margin-top: 50px;
    }
    .login-card {
        background: white; padding: 40px; border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center;
        width: 100%; max-width: 450px; border-top: 6px solid #20B2AA;
    }
    .login-logo { font-size: 60px; margin-bottom: 15px; }
    .login-title { font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px; }
    
    /* Karşı taraf yazıyor efekti */
    .typing-indicator {
        font-style: italic; color: #555; font-size: 12px; margin-bottom: 10px;
        animation: blink 1.5s infinite;
    }
    @keyframes blink { 0% { opacity: .2; } 50% { opacity: 1; } 100% { opacity: .2; } }
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
if 'chat_turn' not in st.session_state: st.session_state.chat_turn = "student" 

# --- 4. MODEL VE FONKSİYONLAR ---
KARA_LISTE = ["siktir", "sik", "amk", "aq", "oç", "piç", "yavşak", "gerizekalı", "salak", "aptal", "mal", "defol", "şerefsiz"]
DOSYA_ADI = "veri_havuzu.xlsx"

GERI_DONUTLER = {
    "Küfür / Hakaret": "Bu mesajda kullanılan dil, saygı sınırlarını aşıyor olabilir. Dijital dünyada güçlü bir iletişimci olmak için nezaket önemlidir. Lütfen mesajını daha yapıcı bir dille yeniden yazar mısın?",
    "Siber Zorbalık": "Bu ifade karşı tarafta üzüntü veya korku yaratabilir. SiberKalkan olarak dijital ayak izinin temiz kalmasını önemsiyoruz. Lütfen bu mesajı gönderme ve ifadelerini yumuşat.",
    "Tehdit": "Tehdit içeren ifadeler hem etik değildir hem de yasal sorunlar doğurabilir. Lütfen öfkeni kontrol et ve barışçıl bir dil kullanmayı dene.",
    "Taciz": "Bu tür ifadeler kişisel sınırları ihlal eder. Lütfen karşındakinin sınırlarına saygı duy.",
    "Genel": "Bu mesaj topluluk kurallarına uygun görünmüyor. Lütfen daha nazik bir ifade kullanmayı dene."
}

# --- PDF İŞLEMLERİ (PEDAGOJİK ALGORİTMA DAHİL) ---
def tr_pdf(text):
    degisim = str.maketrans("ğĞıİşŞçÇöÖüÜ", "gGiIsScCoOuU")
    return text.translate(degisim)

def create_pdf_report(score, history, name="Öğrenci"):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(7, 94, 84)
    pdf.cell(0, 10, tr_pdf("SiberKalkan Veli Bilgilendirme Raporu"), ln=True, align='C')
    pdf.ln(5)
    
    # Tarih
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    tarih = datetime.now().strftime("%d.%m.%Y - %H:%M")
    pdf.cell(0, 10, tr_pdf(f"Öğrenci: {name} | Tarih: {tarih}"), ln=True, align='C')
    pdf.ln(10)
    
    # 1. Puan
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, tr_pdf("1. DIJITAL VATANDASLIK PUANI"), ln=True)
    
    # Bar Chart
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 55, 190, 15, 'F')
    
    if score >= 80: pdf.set_fill_color(76, 175, 80)
    elif score >= 50: pdf.set_fill_color(255, 152, 0)
    else: pdf.set_fill_color(244, 67, 54)
    
    bar_width = (score / 150) * 190
    if bar_width > 190: bar_width = 190
    if bar_width < 0: bar_width = 0
    pdf.rect(10, 55, bar_width, 15, 'F')
    
    pdf.set_y(60)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 5, tr_pdf(f"Puan: {score}"), ln=True, align='C')
    pdf.ln(20)

    # 2. İstatistikler
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, tr_pdf("2. OTURUM ISTATISTIKLERI"), ln=True)
    
    toplam_mesaj = len(history)
    sorunlu_mesaj = sum(1 for h in history if "Normal" not in h['Sonuç'])
    guvenli_mesaj = toplam_mesaj - sorunlu_mesaj
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, tr_pdf(f"- Toplam Islenen Mesaj: {toplam_mesaj}"), ln=True)
    pdf.cell(0, 8, tr_pdf(f"- Guvenli Icerik Sayisi: {guvenli_mesaj}"), ln=True)
    pdf.set_text_color(198, 40, 40)
    pdf.cell(0, 8, tr_pdf(f"- Riskli Girisim / Engellenen: {sorunlu_mesaj}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # 3. Pedagojik
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, tr_pdf("3. PEDAGOJIK DEGERLENDIRME VE TAVSIYE"), ln=True)
    
    risk_orani = sorunlu_mesaj / toplam_mesaj if toplam_mesaj > 0 else 0
    tavsiye = ""
    pdf.set_font("Arial", '', 11)
    
    if risk_orani > 0.30:
        tavsiye = f"Sayin Veli, {name} simulasyon suresince sistem uyarilariyla puan kazanmis olsa bile, SIK SIK (Mesajlarin %{int(risk_orani*100)}'i) zorbalik iceren ifadeler kullanmaya yeltendi. Sistem engelledigi icin puan dusmemis olabilir ancak cocugun 'Zorbalik Egilimi' ve 'Ofke Kontrolu' konusunda ciddi bir rehberlik destegine ihtiyaci var. (Asagidaki engellenenler listesine bakiniz)."
    elif risk_orani > 0:
        if score >= 50:
            tavsiye = f"Sayin Veli, {name} zaman zaman duygusal tepkiler vererek riskli ifadeler kullandi. Sistem uyarisiyla 'Vazgecme' veya 'Duzeltme' davranisi gosterse de, zihninden gecen kelimeler asagidaki listede raporlanmistir. Dijital dil konusunda uyari yapilmasi onerilir."
        else:
            tavsiye = f"Sayin Veli, {name} riskli ifadeler kullandi ve uyarilara ragmen israrci davranislar sergiledi. Dijital empati konusunda desteklenmelidir."
    else:
        tavsiye = f"Sayin Veli, {name} dijital iletisimde son derece saygili, temiz ve ornek bir tutum sergiledi. Hicbir riskli girisimde bulunmadi. Tebrik ediyoruz."
    
    pdf.multi_cell(0, 8, tr_pdf(tavsiye))
    pdf.ln(10)

    # 4. Liste
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, tr_pdf("4. ENGELLENEN ICERIKLER VE GIRISIMLER"), ln=True)
    pdf.set_font("Arial", '', 10)

    riskli_mesajlar = [h for h in history if "Normal" not in h['Sonuç']]

    if riskli_mesajlar:
        pdf.set_text_color(198, 40, 40)
        for msg in riskli_mesajlar:
            temiz_mesaj = str(msg['Metin']).replace("\n", " ")[:60]
            kategori = msg['Sonuç']
            pdf.cell(0, 8, tr_pdf(f"- [{kategori}] {temiz_mesaj}"), ln=True)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, tr_pdf("Bu oturumda hicbir riskli icerik veya girisim tespit edilmemistir."), ln=True)
    
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, tr_pdf("Bu rapor SiberKalkan Yapay Zeka Sistemi tarafindan otomatik olusturulmustur."), align='C')
    
    return pdf.output(dest='S').encode('latin-1')


def kara_liste_kontrolu(metin):
    metin_kucuk = metin.lower()
    for kelime in KARA_LISTE:
        if kelime in metin_kucuk: return True, kelime
    return False, None

def excel_hafiza_kontrolu(metin):
    if os.path.exists(DOSYA_ADI):
        try:
            df = pd.read_excel(DOSYA_ADI)
            bulunan = df[df['Metin'].astype(str).str.lower().str.strip() == metin.lower().strip()]
            if not bulunan.empty:
                son_kayit = bulunan.iloc[-1]
                etiket = son_kayit['Etiket']
                if etiket in ["Siber Zorbalık", "Tehdit", "Küfür / Hakaret", "Taciz"]: return True, etiket
        except: pass
    return False, None

@st.cache_resource
def model_yukle():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "siber_kalkan_modeli")
        
        # 1. YEREL MODEL (Varsa)
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model_type = "local"
        else:
            # 2. BULUT MODEL (GitHub/Streamlit)
            model_name = "savasy/bert-base-turkish-sentiment-cased" 
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
            model_type = "cloud"
        return tokenizer, model, model_type
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None, None, None

tokenizer, model, model_type = model_yukle()

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
# 💾 SAYFA 3: VERİ DÜZENLEME EKRANI
# ==========================================
def show_data_editor():
    st.title("📝 Veri Seti Düzenleme Paneli")
    if not os.path.exists(DOSYA_ADI):
        st.error(f"Henüz bir veri dosyası ({DOSYA_ADI}) bulunmuyor.")
        if st.button("⬅️ Panele Dön"):
            st.session_state.page = 'backend'
            st.rerun()
        return

    try:
        df = pd.read_excel(DOSYA_ADI)
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        return

    edited_df = st.data_editor(
        df,
        num_rows="dynamic", 
        use_container_width=True,
        key="editor",
        hide_index=True,
        column_config={
            "Metin": st.column_config.TextColumn("İfade / Cümle"),
            "Etiket": st.column_config.SelectboxColumn("Kategori", options=["Siber Zorbalık", "Tehdit", "Küfür / Hakaret", "Taciz", "Normal / Güvenli", "Engellendi", "Eğitim-Engellendi"], required=True),
        }
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 GÜNCELLE VE KAYDET", type="primary"):
            try:
                edited_df.to_excel(DOSYA_ADI, index=False)
                st.success("✅ Güncellendi!")
                time.sleep(1)
                st.rerun()
            except: st.error("Hata")
    with col2:
        if st.button("⬅️ PANELE DÖN"):
            st.session_state.page = 'backend'
            st.rerun()

# ==========================================
# 🖥️ SAYFA 1: BACKEND
# ==========================================
def show_backend():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9471/9471031.png", width=100)
        st.title("SiberKalkan v2.2") 
        st.caption("Yönetici Kontrol Paneli")
        st.markdown("---")
        st.session_state.sim_mode = st.radio("Mod Seç:", ("Oyun Modu (Puanlı)", "Eğitim Modu (Katı Kurallı)"))
        if st.button("📲 MOBİL SİMÜLASYON", use_container_width=True): st.session_state.page = 'mobile'; st.rerun()
        st.markdown("---")
        if os.path.exists(DOSYA_ADI):
            with open(DOSYA_ADI, "rb") as f: st.download_button("📥 Veri İndir", f, file_name="SiberKalkan_Data.xlsx")
        
        st.header("🧠 Modeli Eğit")
        input_key = f"train_input_{st.session_state.train_key_counter}"
        egitim_metni = st.text_area("Örnek Cümle:", height=80, key=input_key)
        egitim_etiketi = st.selectbox("Bu cümle nedir?", ["Siber Zorbalık", "Tehdit", "Küfür / Hakaret", "Taciz", "Normal / Güvenli"])
        
        if st.button("EĞİT VE KAYDET"):
            if egitim_metni:
                veriyi_excele_kaydet(egitim_metni, egitim_etiketi, "1.0 (Manuel)", "Kullanıcı (Eğitim Verisi)")
                st.success("Kaydedildi!")
                st.session_state.history.insert(0, {"Metin": egitim_metni, "Sonuç": egitim_etiketi, "Kaynak": "Manuel Eğitim"})
                st.session_state.train_key_counter += 1; st.rerun()

        st.markdown("---")
        if st.button("✏️ VERİ DÜZENLE"): st.session_state.page = 'data_editor'; st.rerun()

    st.subheader("🛡️ SiberKalkan Tehdit Analiz Merkezi")
    user_input = st.text_area("Analiz:", height=100)
    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
        if user_input and model:
            kural, kelime = kara_liste_kontrolu(user_input)
            hafiza, etiket = excel_hafiza_kontrolu(user_input)
            if kural:
                score_neg = 0.99; score_pos = 0.01; karar = f"Yasaklı ({kelime})" 
                sonuc_etiketi = "Küfür / Hakaret"; is_bullying = True
            elif hafiza:
                score_neg = 1.0; score_pos = 0.0; karar = "Hafıza" 
                sonuc_etiketi = etiket; is_bullying = True
            else:
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
                # --- [ÖNEMLİ DÜZELTME] MODEL TÜRÜNE GÖRE SKOR ---
                if model_type == "cloud":
                    # Savasy Modeli: Index 0 = Negatif, Index 1 = Pozitif
                    score_neg = probs[0][0].item() 
                else:
                    # Yerel Model: Genellikle Index 0 = Zorbalık (Varsayılan)
                    score_neg = probs[0][0].item()

                karar = "SiberKalkan AI"; is_bullying = score_neg > 0.60
                sonuc_etiketi = "Siber Zorbalık" if is_bullying else "Normal"
                
            if is_bullying: st.error(f"🚨 TESPİT: {sonuc_etiketi}"); st.progress(score_neg)
            else: st.success("✅ GÜVENLİ"); st.progress(1.0 - score_neg)
            
            st.session_state.history.insert(0, {"Metin": user_input, "Sonuç": sonuc_etiketi, "Kaynak": karar})
            veriyi_excele_kaydet(user_input, sonuc_etiketi, f"{score_neg:.4f}", karar)

    st.markdown("---")
    if st.button("🗑️ Temizle"): st.session_state.history = []; st.rerun()
    for row in st.session_state.history:
        st.text(f"{row['Metin']} -> {row['Sonuç']}")

# ==========================================
# 📱 SAYFA 2: TABLET SİMÜLASYONU
# ==========================================
def show_mobile():
    col_l, col_m, col_r = st.columns([1, 8, 1])
    with col_l:
        if st.button("⬅️ Geri"): st.session_state.page = 'backend'; st.rerun()
        st.markdown("---")
        if st.session_state.history and st.session_state.student_name:
            pdf_data = create_pdf_report(st.session_state.user_score, st.session_state.history, st.session_state.student_name)
            st.download_button("📄 Karne İndir", data=pdf_data, file_name="karnem.pdf", mime="application/pdf", use_container_width=True)
    
    with col_m:
        if not st.session_state.student_name:
            st.markdown("""<div class="login-container"><div class="login-card"><div class="login-logo">🛡️</div><div class="login-title">Giriş Yap</div></div></div>""", unsafe_allow_html=True)
            _, c, _ = st.columns([1,2,1])
            with c:
                with st.form("l"):
                    n = st.text_input("Adın:")
                    if st.form_submit_button("BAŞLA") and n:
                        st.session_state.student_name = n; st.rerun()
            return

        if st.session_state.get('breathing_phase'):
            placeholder = st.empty()
            for i in range(4, 0, -1):
                placeholder.markdown(f"""<div class="tablet-screen-top" style="justify-content:center; align-items:center;"><div class="calm-circle">{i}</div><div class="calm-text">Sakinleş...</div></div>""", unsafe_allow_html=True)
                time.sleep(1.2)
            st.session_state.breathing_phase = False
            st.session_state.alert_active = True
            st.rerun()
            return 

        mode = st.session_state.sim_mode
        is_game = (mode == "Oyun Modu (Puanlı)")
        
        chat_html = ""
        for msg in st.session_state.chat_log:
            role = "msg-incoming" if msg['role'] == 'incoming' else "msg-outgoing"
            chat_html += f"<div class='{role}'>{msg['text']}</div>"
        
        if st.session_state.get('alert_active'):
             chat_html += f"<div class='msg-pending'>{st.session_state.temp_bad_msg}<br><small>⛔ Bekliyor</small></div>"

        st.markdown(f"""<div class="tablet-screen-top"><div class="tablet-header">👤 {st.session_state.student_name} | ⭐ {st.session_state.user_score}</div>{chat_html}</div>""", unsafe_allow_html=True)
        
        if not st.session_state.get('alert_active'):
            if st.session_state.chat_turn == "student":
                with st.form("chat"):
                    c1, c2 = st.columns([4,1])
                    with c1: u_msg = st.text_input("Mesaj:", label_visibility="collapsed")
                    with c2: sub = st.form_submit_button("GÖNDER")
                    
                    if sub and u_msg:
                        kural, k_word = kara_liste_kontrolu(u_msg)
                        hafiza, h_etiket = excel_hafiza_kontrolu(u_msg)
                        bad = False
                        
                        if kural: bad=True; reason=f"Yasaklı: {k_word}"; typ="Küfür"
                        elif hafiza: bad=True; reason=f"Hafıza: {h_etiket}"; typ=h_etiket
                        else:
                            inputs = tokenizer(u_msg, return_tensors="pt", truncation=True, padding=True, max_length=64)
                            outputs = model(**inputs)
                            probs = F.softmax(outputs.logits, dim=1)
                            
                            # --- [ÖNEMLİ DÜZELTME] MOBİL SKORLAMA ---
                            if model_type == "cloud":
                                neg_score = probs[0][0].item() # Cloud: Index 0 = Negatif
                            else:
                                neg_score = probs[0][0].item() # Local: Index 0

                            if neg_score > 0.60: bad=True; reason="Zorbalık"; typ="Siber Zorbalık"
                        
                        if bad:
                            st.session_state.temp_bad_msg = u_msg
                            st.session_state.temp_reason = reason
                            st.session_state.temp_type = typ
                            st.session_state.breathing_phase = True
                            st.rerun()
                        else:
                            st.session_state.chat_log.append({"role": "outgoing", "text": u_msg})
                            if is_game: st.session_state.user_score += 10
                            st.session_state.history.insert(0, {"Metin": u_msg, "Sonuç": "Normal", "Kaynak": "Mobil"})
                            st.session_state.chat_turn = "counterpart"
                            st.rerun()
            else:
                with st.form("cp"):
                    c1, c2 = st.columns([4,1])
                    with c1: cp_msg = st.text_input("Cevap:", label_visibility="collapsed")
                    with c2: sub = st.form_submit_button("CEVAPLA")
                    if sub and cp_msg:
                        st.session_state.chat_log.append({"role": "incoming", "text": cp_msg})
                        st.session_state.chat_turn = "student"
                        st.rerun()
        else:
            with st.form("alert"):
                st.error(f"⚠️ TESPİT EDİLDİ: {st.session_state.temp_reason}")
                if is_game:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("😇 Vazgeç (+50 Puan)"):
                            st.session_state.user_score += 50
                            st.session_state.alert_active = False
                            st.balloons()
                            veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Engellendi", "1.0", "Mobil-Vazgeçti")
                            # [KRİTİK EKLEME] Vazgeçileni Rapora Ekle
                            st.session_state.history.insert(0, {"Metin": st.session_state.temp_bad_msg, "Sonuç": "Engellendi (Vazgecti)", "Kaynak": "Mobil"})
                            time.sleep(1); st.rerun()
                    with c2:
                        if st.form_submit_button("😈 Gönder (-20 Puan)"):
                            st.session_state.user_score -= 20
                            st.session_state.chat_log.append({"role": "outgoing", "text": st.session_state.temp_bad_msg})
                            st.session_state.history.insert(0, {"Metin": st.session_state.temp_bad_msg, "Sonuç": "Zorbalık", "Kaynak": "Mobil"})
                            st.session_state.alert_active = False
                            st.session_state.chat_turn = "counterpart"
                            st.rerun()
                else:
                    st.info("Eğitim Modu: Mesajını düzeltmelisin.")
                    if st.form_submit_button("✍️ Düzelt"):
                        st.session_state.alert_active = False
                        veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Engellendi", "1.0", "Eğitim")
                        # [KRİTİK EKLEME] Eğitimi Rapora Ekle
                        st.session_state.history.insert(0, {"Metin": st.session_state.temp_bad_msg, "Sonuç": "Engellendi (Eğitim)", "Kaynak": "Mobil"})
                        st.rerun()

# --- YÖNLENDİRME ---
if st.session_state.page == 'backend': show_backend()
elif st.session_state.page == 'mobile': show_mobile()
elif st.session_state.page == 'data_editor': show_data_editor()
