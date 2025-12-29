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

# --- 2. CSS TASARIMI (ORİJİNAL YAPI KORUNDU) ---
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

# --- PDF İŞLEMLERİ (AKILLI RAPORLAMA EKLENDİ) ---
def tr_pdf(text):
    """PDF için Türkçe karakter düzeltmesi"""
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
    
    # Bar Chart Arkaplan
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 55, 190, 15, 'F')
    
    # Bar Chart Dolgu
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
    pdf.cell(0, 8, tr_pdf(f"- Engellenen Zorbalik Girisimi: {sorunlu_mesaj}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # --- 3. PEDAGOJİK DEĞERLENDİRME (AKILLI ALGORİTMA) ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, tr_pdf("3. PEDAGOJIK DEGERLENDIRME VE TAVSIYE"), ln=True)
    
    # Yeni Mantık: Risk Oranına Göre Karar Ver
    risk_orani = sorunlu_mesaj / toplam_mesaj if toplam_mesaj > 0 else 0
    
    tavsiye = ""
    pdf.set_font("Arial", '', 11)
    
    # Durum 1: Yüksek Risk (%30'dan fazla girişim), Puan yüksek olsa BİLE uyar.
    if risk_orani > 0.30:
        tavsiye = f"Sayin Veli, {name} simulasyon suresince sistem uyarilariyla puan kazanmis olsa bile, SIK SIK (Mesajlarin %{int(risk_orani*100)}'i) zorbalik iceren ifadeler kullanmaya yeltendi. Sistem engelledigi icin puan dusmemis olabilir ancak cocugun 'Zorbalik Egilimi' ve 'Ofke Kontrolu' konusunda ciddi bir rehberlik destegine ihtiyaci var."
    
    # Durum 2: Orta Risk (Arada denemiş, vazgeçmiş)
    elif risk_orani > 0:
        if score >= 50:
            tavsiye = f"Sayin Veli, {name} zaman zaman duygusal tepkiler vererek riskli ifadeler kullandi. Ancak sistemin uyarilarini dikkate alip 'Vazgecme' davranisi gosterdi ve kendini duzeltti. Bu, dijital farkindaliginin gelismekte oldugunu gosteriyor ancak takip edilmelidir."
        else:
            tavsiye = f"Sayin Veli, {name} riskli ifadeler kullandi ve uyarilara ragmen yeterli duzeltme davranisi gostermedigi icin puani dustu. Dijital empati konusunda desteklenmelidir."
            
    # Durum 3: Temiz (Hiç girişimi yok)
    else:
        tavsiye = f"Sayin Veli, {name} dijital iletisimde son derece saygili, temiz ve ornek bir tutum sergiledi. Hicbir riskli girisimde bulunmadi. Tebrik ediyoruz."
    
    pdf.multi_cell(0, 8, tr_pdf(tavsiye))
    pdf.ln(10)

    # --- 4. ENGELLENEN MESAJLAR LİSTESİ ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, tr_pdf("4. ENGELLENEN VE RISKLİ ICERIKLER"), ln=True)
    pdf.set_font("Arial", '', 10)

    riskli_mesajlar = [h for h in history if "Normal" not in h['Sonuç']]

    if riskli_mesajlar:
        pdf.set_text_color(198, 40, 40) # Kırmızı Renk
        for msg in riskli_mesajlar:
            # Mesajı biraz kısaltalım taşmasın
            temiz_mesaj = str(msg['Metin']).replace("\n", " ")[:60]
            kategori = msg['Sonuç']
            pdf.cell(0, 8, tr_pdf(f"- [{kategori}] {temiz_mesaj}"), ln=True)
    else:
        pdf.set_text_color(0, 128, 0) # Yeşil Renk
        pdf.cell(0, 8, tr_pdf("Bu oturumda hicbir riskli icerik tespit edilmemistir. Tebrikler!"), ln=True)
    
    # Alt Bilgi
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
# 💾 SAYFA 3: VERİ DÜZENLEME EKRANI
# ==========================================
def show_data_editor():
    st.title("📝 Veri Seti Düzenleme Paneli")
    st.info("Bu ekranda veri tabanındaki kelimeleri silebilir, kategorilerini değiştirebilir veya yeni veri ekleyebilirsiniz.")

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
            "Metin": st.column_config.TextColumn("İfade / Cümle", help="Zorbalık içeren veya normal metin"),
            "Etiket": st.column_config.SelectboxColumn("Kategori", options=["Siber Zorbalık", "Tehdit", "Küfür / Hakaret", "Taciz", "Normal / Güvenli", "Engellendi", "Eğitim-Engellendi"], required=True),
            "Kaynak": st.column_config.TextColumn("Veri Kaynağı", disabled=True),
            "AI_Skoru": st.column_config.TextColumn("Skor", disabled=True),
            "Tarih": st.column_config.TextColumn("Tarih", disabled=True)
        }
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 GÜNCELLE VE KAYDET", type="primary"):
            try:
                edited_df.to_excel(DOSYA_ADI, index=False)
                st.success("✅ Veri seti başarıyla güncellendi!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Kaydetme hatası: {e}")
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
        st.title("SiberKalkan v1.0") 
        st.caption("Yönetici Kontrol Paneli")
        
        st.markdown("---")
        st.subheader("⚙️ Simülasyon Ayarı")
        mod_secimi = st.radio(
            "Öğrenci Ekranı Modu:",
            ("Oyun Modu (Puanlı)", "Eğitim Modu (Katı Kurallı)")
        )
        st.session_state.sim_mode = mod_secimi
        
        if st.button("📲 MOBİL SİMÜLASYON", use_container_width=True):
            st.session_state.page = 'mobile'; st.rerun()
            
        st.markdown("---")
        if os.path.exists(DOSYA_ADI):
            with open(DOSYA_ADI, "rb") as f: st.download_button("📥 Veri Setini İndir", f, file_name="SiberKalkan_Data.xlsx")
        
        st.header("🧠 Modeli Eğit")
        st.info("AI hata yaparsa buradan doğrusunu öğretin.")
        input_key = f"train_input_{st.session_state.train_key_counter}"
        egitim_metni = st.text_area("Örnek Cümle:", placeholder="Kelime giriniz...", height=80, key=input_key)
        egitim_etiketi = st.selectbox("Bu cümle nedir?", ["Siber Zorbalık", "Tehdit", "Küfür / Hakaret", "Taciz", "Normal / Güvenli"])
        
        if st.button("EĞİT VE KAYDET"):
            if egitim_metni:
                veriyi_excele_kaydet(egitim_metni, egitim_etiketi, "1.0 (Manuel)", "Kullanıcı (Eğitim Verisi)")
                st.success("Veri hafızaya alındı! ✅")
                st.session_state.history.insert(0, {"Metin": egitim_metni, "Sonuç": egitim_etiketi, "Kaynak": "Manuel Eğitim"})
                st.session_state.train_key_counter += 1; st.rerun()
            else: st.warning("Metin girmeyi unuttunuz.")

        st.markdown("---")
        if st.button("✏️ VERİ SETİNİ DÜZENLE"):
            st.session_state.page = 'data_editor'
            st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1: st.markdown("## 🛡️ SiberKalkan Tehdit Analiz Merkezi")
    with col2: st.success("🟢 Sistem Aktif")
    col_input, col_result = st.columns([1, 1], gap="medium")
    with col_input:
        user_input = st.text_area("Analiz edilecek mesaj:", height=150, placeholder="Örn: Buraya şüpheli bir metin girin...")
        analyze_btn = st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True)
    with col_result:
        if analyze_btn and user_input and model:
            kural_ihlali, yakalanan_kelime = kara_liste_kontrolu(user_input)
            hafiza_ihlali, hafiza_etiketi = excel_hafiza_kontrolu(user_input)
            if kural_ihlali:
                score_neg = 0.99; score_pos = 0.01; karar_kaynagi = f"Güvenlik Protokolü ({yakalanan_kelime})" 
                sonuc_etiketi = "Küfür / Hakaret"; is_bullying = True
            elif hafiza_ihlali:
                score_neg = 1.0; score_pos = 0.0; karar_kaynagi = "Öğrenilmiş Hafıza (Excel)" 
                sonuc_etiketi = hafiza_etiketi; is_bullying = True
            else:
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                score_neg = probs[0][0].item(); score_pos = probs[0][1].item()
                karar_kaynagi = "SiberKalkan AI"; is_bullying = score_neg > 0.60
                sonuc_etiketi = "Siber Zorbalık" if is_bullying else "Normal / Güvenli"
            st.subheader("📊 Analiz Raporu")
            if is_bullying: st.error(f"🚨 **TESPİT EDİLDİ: {sonuc_etiketi.upper()}**"); st.progress(score_neg)
            else: st.success(f"✅ **GÜVENLİ İÇERİK**"); st.progress(score_pos)
            veriyi_excele_kaydet(user_input, sonuc_etiketi, f"{score_neg:.4f}", karar_kaynagi)
            st.session_state.history.insert(0, {"Metin": user_input, "Sonuç": sonuc_etiketi, "Kaynak": karar_kaynagi})
    st.markdown("---")
    c_head, c_clear = st.columns([4,1])
    with c_head: st.subheader("📝 Son İşlemler (Oturum Geçmişi)")
    with c_clear: 
        if st.button("🗑️ Tümünü Temizle"): st.session_state.history = []; st.rerun()
    if st.session_state.history:
        for i, row in enumerate(st.session_state.history):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.text(row['Metin'][:40])
            if "Normal" in row['Sonuç']: c2.success(row['Sonuç'])
            else: c2.error(row['Sonuç'])
            c3.caption(row['Kaynak'])
            if c4.button("Sil", key=f"del_{i}"): del st.session_state.history[i]; st.rerun()
    else: st.info("Veri yok.")

# ==========================================
# 📱 SAYFA 2: TABLET SİMÜLASYONU
# ==========================================
def show_mobile():
    col_l, col_m, col_r = st.columns([1, 8, 1])
    with col_l:
        if st.button("⬅️ Panele Dön"): st.session_state.page = 'backend'; st.rerun()
        
        st.markdown("---")
        # PDF BUTONU (İsim varsa kullan)
        if st.session_state.history and st.session_state.student_name:
            pdf_data = create_pdf_report(st.session_state.user_score, st.session_state.history, st.session_state.student_name)
            st.download_button(
                label="📄 Veli Karnesi",
                data=pdf_data,
                file_name="SiberKalkan_Veli_Raporu.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col_m:
        # --- GİRİŞ EKRANI (PROFESYONEL - GÜNCELLENDİ) ---
        if not st.session_state.student_name:
            st.markdown("""
            <div class="login-container">
                <div class="login-card">
                    <div class="login-logo">🛡️</div>
                    <div class="login-title">SiberKalkan'a Hoş Geldin</div>
                    <div style="color: #666; font-size: 14px; margin-bottom: 20px;">
                        Simülasyonu başlatmak için lütfen adınızı giriniz.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Formu ortalamak için kolon hilesi (Görsel bütünlük için)
            _, col_form, _ = st.columns([1, 2, 1])
            with col_form:
                with st.form("login_form"):
                    name_input = st.text_input("Adın Soyadın:", placeholder="Örn: Ali Veli")
                    if st.form_submit_button("SİMÜLASYONU BAŞLAT ▶", use_container_width=True):
                        if name_input:
                            st.session_state.student_name = name_input
                            st.rerun()
            return # Giriş yapılmadıysa aşağıyı gösterme!

        # --- SAKİNLEŞME MODU (ORİJİNAL KOD) ---
        if st.session_state.get('breathing_phase'):
            placeholder = st.empty()
            # 6 Saniye Geri Sayım
            for i in range(4, 0, -1):
                placeholder.markdown(f"""
                <div class="tablet-screen-top" style="align-items: center; justify-content: center; background-color: #E1F5FE;">
                    <div class="calm-text">🧘‍♂️ Çok Öfkeli Görünüyorsun...</div>
                    <div class="calm-subtext">Mesajını analiz etmeden önce derin bir nefes al 🧘</div>
                    <div class="calm-circle">{i}</div>
                    <div class="calm-subtext">Sakinleşiyoruz...</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1.2) 
            
            placeholder.empty()
            st.session_state.breathing_phase = False
            st.session_state.alert_active = True
            st.rerun()
            return 

        mode = st.session_state.sim_mode
        is_game_mode = (mode == "Oyun Modu (Puanlı)")
        
        # --- CHAT İÇERİĞİ VE "YAZIYOR..." EFEKTİ (TABLET İÇİNE GÖMÜLDÜ) ---
        chat_html = ""
        for msg in st.session_state.chat_log:
            role_class = "msg-incoming" if msg['role'] == 'incoming' else "msg-outgoing"
            chat_html += f"<div class='{role_class}'>{msg['text']}</div>"
        
        # Karşı Taraf Yazıyor Göstergesi (Tabletin içinde!)
        if not st.session_state.get('alert_active') and st.session_state.chat_turn == "counterpart":
            chat_html += f"""
            <div style='clear:both;'></div>
            <div class='typing-indicator'>💬 Karşı taraf yazıyor...</div>
            """

        if st.session_state.get('alert_active'):
             chat_html += f"""
             <div style='clear:both;'></div>
             <div class='msg-pending'>
                {st.session_state.temp_bad_msg} <br>
                <small>⛔ Onay Bekliyor</small>
             </div>
             """

        score_display = f"<span class='score-board'>⭐ {st.session_state.user_score}</span>" if is_game_mode else ""
        
        # HEADER (İSİM EKLENDİ)
        st.markdown(f"""
        <div class="tablet-screen-top">
            <div class="tablet-header">
                👤 {st.session_state.student_name} | SiberKalkan {score_display}
            </div>
            {chat_html}
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.get('alert_active'):
            
            # --- SOHBET SIRASI KİMDE? ---
            if st.session_state.chat_turn == "student":
                # SIRA ÖĞRENCİDE
                with st.form("chat_form_input", clear_on_submit=True):
                    c_in, c_btn = st.columns([4, 1])
                    with c_in:
                        user_msg = st.text_input("Mesajın:", placeholder="Bir şeyler yaz...", label_visibility="collapsed")
                    with c_btn:
                        submitted = st.form_submit_button("GÖNDER", use_container_width=True)
                    
                    if submitted and user_msg:
                        is_bullying = False; reason = ""
                        violation_type = "Genel"
                        
                        kural, kelime = kara_liste_kontrolu(user_msg)
                        hafiza, etiket = excel_hafiza_kontrolu(user_msg)
                        
                        if kural: 
                            is_bullying=True; reason=f"Yasaklı Kelime: {kelime}"; violation_type = "Küfür / Hakaret"
                        elif hafiza: 
                            is_bullying=True; reason=f"Tespit Edilen: {etiket}"; violation_type = etiket
                        else:
                            inputs = tokenizer(user_msg, return_tensors="pt", truncation=True, padding=True, max_length=64)
                            outputs = model(**inputs)
                            if F.softmax(outputs.logits, dim=1)[0][0].item() > 0.60: 
                                is_bullying=True; reason="Saldırgan Dil"; violation_type = "Siber Zorbalık"
                        
                        if is_bullying:
                            st.session_state.temp_bad_msg = user_msg
                            st.session_state.temp_reason = reason
                            st.session_state.temp_type = violation_type
                            # Sakinleşme modunu tetikle
                            st.session_state.breathing_phase = True
                            st.rerun()
                        else:
                            st.session_state.chat_log.append({"role": "outgoing", "text": user_msg})
                            if is_game_mode: st.session_state.user_score += 10 
                            st.session_state.history.insert(0, {"Metin": user_msg, "Sonuç": "Normal", "Kaynak": "Mobil"})
                            # SIRA KARŞIYA GEÇTİ
                            st.session_state.chat_turn = "counterpart"
                            st.rerun()
            
            else:
                # SIRA KARŞI TARAFTA (Uyarıyı kaldırdık, sadece kutu kaldı)
                with st.form("counterpart_form", clear_on_submit=True):
                    c_in_cp, c_btn_cp = st.columns([4, 1])
                    with c_in_cp:
                        cp_msg = st.text_input("Senaryo Cevabı:", placeholder="Karşı tarafın cevabını girin...", label_visibility="collapsed")
                    with c_btn_cp:
                        submitted_cp = st.form_submit_button("CEVAPLA", use_container_width=True)
                    
                    if submitted_cp and cp_msg:
                        st.session_state.chat_log.append({"role": "incoming", "text": cp_msg})
                        # SIRA TEKRAR ÖĞRENCİYE GEÇTİ
                        st.session_state.chat_turn = "student"
                        st.rerun()

        else:
            # UYARI EKRANI (ORİJİNAL)
            with st.form("chat_form_alert"):
                if is_game_mode:
                    st.markdown(f"""
                    <div class="tablet-alert-box">
                        ⚠️ DUR! SiberKalkan Tehdit Algıladı: {st.session_state.temp_reason}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("😇 Vazgeç (+50 Puan)", use_container_width=True):
                            st.session_state.user_score += 50
                            st.session_state.alert_active = False
                            st.balloons(); veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Engellendi", "1.0", "Mobil-Vazgeçti")
                            time.sleep(1.0); st.rerun()
                    with c2:
                        if st.form_submit_button("😈 Gönder (-20 Puan)", use_container_width=True):
                            st.session_state.user_score -= 20
                            st.session_state.chat_log.append({"role": "outgoing", "text": st.session_state.temp_bad_msg})
                            st.session_state.history.insert(0, {"Metin": st.session_state.temp_bad_msg, "Sonuç": "Zorbalık", "Kaynak": "Mobil-İnat"})
                            st.session_state.alert_active = False; 
                            # Gönderse bile sıra karşıya geçsin
                            st.session_state.chat_turn = "counterpart"
                            st.rerun()
                
                else:
                    feedback_msg = GERI_DONUTLER.get(st.session_state.temp_type, GERI_DONUTLER["Genel"])
                    st.markdown(f"""
                    <div class="tablet-alert-box" style="border-color: #4db6ac; color: #00695c; background-color: #e0f2f1;">
                        🎓 SİBERKALKAN REHBERLİK SERVİSİ
                        <div style="font-weight: normal; margin-top: 5px; color: #333;">
                            "{feedback_msg}"
                        </div>
                    </div>
                    <div class="guide-message">Mesajını düzeltmek için aşağıdaki butona tıkla.</div>
                    """, unsafe_allow_html=True)
                    
                    if st.form_submit_button("✍️ Anladım, Mesajımı Düzelteceğim", use_container_width=True):
                        st.session_state.alert_active = False
                        veriyi_excele_kaydet(st.session_state.temp_bad_msg, "Eğitim-Engellendi", "1.0", "Mobil-EğitimModu")
                        st.rerun()

# --- ANA YÖNLENDİRİCİ ---
if st.session_state.page == 'backend': 
    show_backend()
elif st.session_state.page == 'mobile': 
    show_mobile()
elif st.session_state.page == 'data_editor': 
    show_data_editor()