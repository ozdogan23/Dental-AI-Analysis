import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd
import os

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="Diş Röntgeni Anomali Tespiti",
    page_icon="🦷",
    layout="wide"
)

# Başlık ve Alt Başlık
st.title("🦷 Diş Röntgeni Anomali Tespiti")
st.write("Eğitilmiş YOLOv8m modeli kullanarak bir diş röntgenindeki anomalileri (`caries` ve `trapezoid`) tespit edin.")


# Modelin her seferinde yeniden yüklenmesini engelleyen fonksiyon
@st.cache_resource
def load_model(model_path):
    """
    YOLO modelini yükler.
    """
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.error(
            "Lütfen Kaggle'dan indirdiğiniz 'best.pt' dosyasının bu kodla ('app.py') aynı klasörde olduğundan emin olun.")
        return None
    model = YOLO(model_path)
    return model


# Modeli yükle
model = load_model('best.pt')

# Eğer model başarıyla yüklendiyse arayüzü göster
if model is not None:
    # Kullanıcının resim yüklemesi için bir alan oluştur
    uploaded_file = st.file_uploader("Analiz için bir röntgen görüntüsü yükleyin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Yüklenen dosyayı bir resim objesi olarak aç
        image = Image.open(uploaded_file)

        # Arayüzü iki sütuna böl
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Orijinal Görüntü")
            # <-- DEĞİŞİKLİK 1: Resim genişliği ayarlandı. 'use_column_width' kaldırıldı.
            st.image(image, caption="Yüklenen Röntgen", width=600)

            # "Hesaplanıyor..." mesajı göstererek tahmin işlemini başlat
        with st.spinner('🤖 Model anomalileri tespit ediyor, lütfen bekleyin...'):
            results = model(image, verbose=False)
            result_image_plotted = results[0].plot()
            result_image_rgb = cv2.cvtColor(result_image_plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Model Tahmin Sonuçları")
            # <-- DEĞİŞİKLİK 1: Sonuç resminin genişliği de ayarlandı.
            st.image(result_image_rgb, caption="Tespit Edilen Anomaliler", width=600)

        # Tespit edilen nesneler hakkında detaylı bilgi içeren bir tablo oluştur
        st.subheader("Tespit Detayları")

        detected_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_objects.append([class_name, f"{confidence:.2f}"])

        # <-- DEĞİŞİKLİK 2: Bilgilendirme mesajları eklendi.
        if detected_objects:
            df = pd.DataFrame(detected_objects, columns=["Tespit Edilen Sınıf", "Güven Skoru"])
            st.dataframe(df, use_container_width=True)
            # Anomali bulunduğunda gösterilecek şık bilgilendirme kutusu
            st.info(
                "ℹ️ Bilgilendirme: Yukarıda işaretlenen anomaliler dışındaki bölgeler sağlıklı olarak kabul edilmektedir.")
        else:
            # Anomali bulunmadığında gösterilecek şık başarı mesajı
            st.success(
                "✅ Model bu resimde herhangi bir anomali (caries veya trapezoid) tespit etmedi. Görüntü 'sağlıklı' olarak değerlendirilmiştir.")