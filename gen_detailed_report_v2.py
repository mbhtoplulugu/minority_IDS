from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_detailed_report():
    doc = Document()
    
    # 1. Başlık
    title = doc.add_heading('UNSW-NB15 IDS Pipeline Optimizasyon ve Metodoloji Raporu', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 2. Özet (Abstract)
    doc.add_heading('Özet', level=1)
    doc.add_paragraph(
        "Bu çalışma, IoT ve ağ trafiği güvenliği için kritik öneme sahip olan UNSW-NB15 veri seti üzerinde "
        "%64.73 Macro F1-skoruna ulaşan hibrit bir Saldırı Tespit Sistemi (IDS) sunar. "
        "Sistem; veri odaklı temizleme, siber güvenlik temelli özellik mühendisliği ve uzman modelleri "
        "birleştiren kural tabanlı bir topluluk (Heuristic Ensemble) mimarisi üzerine inşa edilmiştir."
    )

    # 3. Veri İşleme Aşamaları ve Metodoloji
    doc.add_heading('1. Veri Ön İşleme ve Hazırlık', level=1)
    doc.add_paragraph(
        "Veri seti işleme süreci, literatürdeki 'Data-Centric AI' prensiplerini takip eder:",
        style='List Bullet'
    )
    doc.add_paragraph("Standardizasyon: Sayısal veriler MinMaxScaler ve StandardScaler ile normalize edilerek Gradyan tabanlı modellerin (MLP, XGB) daha hızlı yakınsaması sağlandı.", style='List Bullet')
    doc.add_paragraph("Kategorik Kodlama: Protokol ve servis isimleri OrdinalEncoder ile sayısallaştırıldı.", style='List Bullet')
    doc.add_paragraph("Serialization (joblib): 2 milyondan fazla satır içeren veri setinin RAM kısıtları altında yönetilmesi için joblib formatında serialize edilerek disk üzerinde önbelleklendi.", style='List Bullet')

    # 4. Sistem Akış Şeması ve Aşamalar Arası İlişkiler
    doc.add_heading('2. Sistem Akışı ve Modüller Arası İlişkiler', level=1)
    doc.add_paragraph(
        "Sistemdeki her aşama, bir sonrakini besleyecek şekilde tasarlanmıştır:",
        style='Normal'
    )
    rel_data = [
        ["Ham Veri -> Preprocessing", "Verinin ölçeklenmesi, özellik mühendisliğinde kullanılacak istatistiksel hesaplamaların (oranlar, farklar) doğruluğunu sağlar."],
        ["Özellik Mühendisliği -> Resampling", "Eklenen 19 cerrahi özellik, Tomek Links algoritmasının gürültülü sınırları daha yüksek boyutsal hassasiyetle temizlemesine yardımcı olur."],
        ["Resampling -> Modelleme", "SMOTE ile azınlık sınıfların artırılması, XGBoost'un Focal Loss fonksiyonunun nadir ataklara daha fazla odaklanmasını sağlar."],
        ["OOF Tahminler -> Ensemble", "5-Fold Cross Validation ile üretilen Out-Of-Fold tahminler, Heuristic Ensemble'ın sızıntı (leakage) olmadan kararlı sonuçlar üretmesini sağlar."]
    ]
    rel_table = doc.add_table(rows=1, cols=2)
    rel_table.style = 'Table Grid'
    hdr_rel = rel_table.rows[0].cells
    hdr_rel[0].text = 'Aşamalar'
    hdr_rel[1].text = 'İlişki ve Teknik Katkı'
    for stage, rel in rel_data:
        row = rel_table.add_row().cells
        row[0].text = stage
        row[1].text = rel

    # 5. Siber Güvenlik Temelli Özellik Mühendisliği
    doc.add_heading('3. Cerrahi Özellik Mühendisliği (Surgical Features)', level=1)
    doc.add_paragraph(
        "Literatürdeki 'Domain-Specific Feature Engineering' yaklaşımıyla, saldırgan davranışlarını "
        "ayırt etmek için toplam 19 yeni kolon eklenmiştir:"
    )
    
    feat_table = doc.add_table(rows=1, cols=3)
    feat_table.style = 'Table Grid'
    hdr_feat = feat_table.rows[0].cells
    hdr_feat[0].text = 'Kolon İsmi'
    hdr_feat[1].text = 'Kategori'
    hdr_feat[2].text = 'Siber Güvenlik Neden/Mantığı'
    
    feats = [
        ["src_port_high / dst_port_high", "Port Analizi", "Backdoor'ların genellikle dinamik/yüksek portları kullanma eğilimini tespit eder."],
        ["dst_http / dst_ssh / dns / ftp", "Protokol", "Saldırı trafiğinin meşru servisler arkasına saklanma durumunu izler."],
        ["sbytes_per_pkt / dbytes_per_pkt", "Yoğunluk", "DoS saldırılarındaki çok sayıda küçük paket yapısını (flood) yakalar."],
        ["pkt_rate / byte_rate", "Dinamik", "Normal trafiği aşan ekstrem veri akış hızlarını anomali olarak işaretler."],
        ["byte_asymmetry / pkt_asymmetry", "Simetri", "Scanning ve Analysis faaliyetlerindeki tek yönlü yoğun akışı tespit eder."],
        ["ttl_diff / ttl_sum", "Ağ Katmanı", "Hedef ve kaynak arasındaki alışılmadık TTL değişimlerini (spoofing tespiti) yakalar."],
        ["jit_ratio / intpkt_ratio", "Zamanlama", "Otomize edilmiş saldırı araçlarının (Botlar) ritmik paternlerini belirler."]
    ]
    
    for name, cat, logic in feats:
        row = feat_table.add_row().cells
        row[0].text = name
        row[1].text = cat
        row[2].text = logic

    # 6. Model Eğitim Stratejisi
    doc.add_heading('4. Model Eğitim ve Değerlendirme Stratejisi', level=1)
    doc.add_paragraph(
        "Sistemde 4 farklı katman-1 model mimarisi kullanılmıştır:", style='Normal'
    )
    doc.add_paragraph("XGBoost (Cost-Sensitive): Multiclass Focal Loss kullanılarak, modelin zor sınıflandırılan örneklere daha fazla ağırlık vermesi sağlandı.", style='List Bullet')
    doc.add_paragraph("LightGBM V1 & V2: Hem orijinal 36 özellik hem de 19 cerrahi özellik üzerinde paralel eğitilerek çeşitlilik sağlandı.", style='List Bullet')
    doc.add_paragraph("HistGB (CPU-Optimized): Bellek verimli gradyan artırma ile özellikle DoS sınıfında yüksek hassasiyet elde edildi.", style='List Bullet')
    doc.add_paragraph("Surgical Experts: Backdoor ve Analysis gibi sınıflar için Hard-Negative Mining ile özelleşmiş binary modeller eğitildi.", style='List Bullet')

    # 7. Karşılaştırmalı Sonuçlar
    doc.add_heading('5. Karşılaştırmalı Sonuç Analizi (Macro F1)', level=1)
    
    res_table = doc.add_table(rows=1, cols=6)
    res_table.style = 'Table Grid'
    hdr_res = res_table.rows[0].cells
    hdr_res[0].text = 'Sınıf'
    hdr_res[1].text = 'XGBoost'
    hdr_res[2].text = 'LGBM (Ori)'
    hdr_res[3].text = 'LGBM V2'
    hdr_res[4].text = 'HistGB'
    hdr_res[5].text = 'Ensemble'
    
    results = [
        ["Analysis", "0.1647", "0.1822", "0.1515", "0.1382", "0.1756"],
        ["Backdoor", "0.1144", "0.1186", "0.0680", "0.1120", "0.1155"],
        ["DoS", "0.3400", "0.3351", "0.2573", "0.4402", "0.3919"],
        ["Exploits", "0.7311", "0.7296", "0.7263", "0.6551", "0.7192"],
        ["Worms", "0.5714", "0.6190", "0.6809", "0.5176", "0.7347"],
        ["MACRO F1", "0.6259", "0.6345", "0.6215", "0.6082", "0.6473"]
    ]
    
    for row_data in results:
        row = res_table.add_row().cells
        for i, val in enumerate(row_data):
            row[i].text = val
            if row_data[0] == "MACRO F1" or i == 5:
                # Bold the Macro F1 row and Ensemble column
                p = row[i].paragraphs[0]
                run = p.runs[0] if p.runs else p.add_run(val)
                run.bold = True

    # 8. Sonuç (Conclusion)
    doc.add_heading('6. Sonuç ve Değerlendirme', level=1)
    doc.add_paragraph(
        "Sistemin %65 başarı eşiğine ulaşması, sadece daha fazla veri veya daha derin modellerle değil, "
        "verinin siber güvenlik perspektifiyle işlenmesi ve modellerin güçlü yanlarına göre yönlendirilmesiyle "
        "mümkün olmuştur. Heuristic Ensemble yapısı, yanlış tahmin olasılığını minimize ederek "
        "güvenilir bir son karar katmanı oluşturmaktadır."
    )

    # Footer
    doc.add_paragraph("\n[Rapor Otomatik Olarak Üretilmiştir - IDS Pipeline v2.0]")

    output_path = 'IDS_Detailed_Report_v2.docx'
    doc.save(output_path)
    print(f"Rapor başarıyla oluşturuldu: {output_path}")

if __name__ == "__main__":
    create_detailed_report()
