"""
generate_feature_doc.py
Muhendislik edilen ozniteliklerin detayli Word dokumani olusturur.
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Sayfa kenarlari ──────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Cm(21)
section.page_height = Cm(29.7)
section.left_margin = section.right_margin = Cm(2.5)
section.top_margin  = section.bottom_margin = Cm(2.5)

# ── Stil yardimcilari ────────────────────────────────────────────────────────
def set_font(run, name="Times New Roman", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    sz = {1: 14, 2: 12, 3: 11}[level]
    bold = True
    set_font(run, size=sz, bold=bold, color=(0,70,127))
    p.paragraph_format.space_before = Pt(14 if level==1 else 10)
    p.paragraph_format.space_after  = Pt(4)
    return p

def add_para(doc, text="", bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_font(r, bold=True)
    if text:
        r = p.add_run(text)
        set_font(r, italic=italic)
    return p

def add_formula(doc, formula_text):
    """Formulu italik, girintili, ortalanmis paragraf olarak ekle."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    r = p.add_run(formula_text)
    set_font(r, name="Courier New", size=10, italic=True)
    return p

def add_bullet(doc, text, bold_part=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_part:
        r = p.add_run(bold_part)
        set_font(r, bold=True)
        r2 = p.add_run(text)
        set_font(r2)
    else:
        r = p.add_run(text)
        set_font(r)
    return p

def shade_cell(cell, hex_color="D9E1F2"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def add_table(doc, headers, rows, col_widths=None, header_color="1F4E79"):
    tbl = doc.add_table(rows=1+len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Baslik satiri
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade_cell(cell, "1F4E79")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_font(r, bold=True, size=9, color=(255,255,255))

    # Veri satirlari
    for i, row in enumerate(rows):
        bg = "EBF3FB" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row):
            cell = tbl.rows[i+1].cells[j]
            shade_cell(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            set_font(r, size=8.5)

    if col_widths:
        for i, row in enumerate(tbl.rows):
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    return tbl


# ════════════════════════════════════════════════════════════════════════════
# KAPAK
# ════════════════════════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(36)
r = p.add_run("Mühendislik Edilen Öznitelikler")
set_font(r, size=18, bold=True, color=(0,70,127))

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Formüller, Semboller, Hesaplama Yöntemi ve Literatür")
set_font(r2, size=13, italic=True, color=(80,80,80))

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("Ağ Trafiği Tabanlı Saldırı Tespit Sistemi (IDS)")
set_font(r3, size=11, color=(100,100,100))

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# GIRIS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "1. Giriş", 1)
add_para(doc, (
    "Bu belgede, saldırı tespit sisteminin özellik mühendisliği aşamasında ham "
    "ağ akışı istatistiklerine ek olarak türetilen 28 benzersiz öznitelik "
    "detaylı biçimde açıklanmaktadır. Her öznitelik için: (i) hesaplamada "
    "kullanılan ham veri sütunları ve semboller, (ii) matematiksel formül, "
    "(iii) formülde geçen sembollerin tanımı, (iv) hangi saldırı sınıflarını "
    "ayırt etmeye katkı sağladığı, (v) destekleyici literatür kaynakları "
    "verilmiştir. Öznitelikler dört ana grupta sunulmaktadır: İstatistiksel "
    "Anomali Göstergeleri (A), Anomali Skoru (B), Alan Bilgisi Özellikleri (C) "
    "ve Derin Siber Güvenlik Özellikleri (D)."
))

add_heading(doc, "2. Sembol Sözlüğü", 1)
add_para(doc, (
    "Aşağıdaki tablo, formüllerde kullanılan temel sembollerin karşılık "
    "geldiği ham veri sütunlarını göstermektedir."
))

sym_headers = ["Sembol", "Ham Sütun Adı", "Açıklama"]
sym_rows = [
    ["sbytes",  "sbytes",  "Kaynak → Hedef yönünde iletilen toplam byte sayısı"],
    ["dbytes",  "dbytes",  "Hedef → Kaynak yönünde iletilen toplam byte sayısı"],
    ["spkts",   "spkts",   "Kaynak → Hedef yönünde gönderilen toplam paket sayısı"],
    ["dpkts",   "dpkts",   "Hedef → Kaynak yönünde gönderilen toplam paket sayısı"],
    ["dur",     "dur",     "Bağlantı/akış süresi (saniye)"],
    ["dsport",  "dsport",  "Hedef (destination) port numarası"],
    ["sport",   "sport",   "Kaynak (source) port numarası"],
    ["sttl",    "sttl",    "Kaynak paketlerinin TTL (Time-To-Live) değeri"],
    ["dttl",    "dttl",    "Hedef paketlerinin TTL değeri"],
    ["sloss",   "sloss",   "Kaynak tarafında tespit edilen kayıp paket sayısı"],
    ["dloss",   "dloss",   "Hedef tarafında tespit edilen kayıp paket sayısı"],
    ["tcprtt",  "tcprtt",  "TCP round-trip süresi (saniye)"],
    ["synack",  "synack",  "SYN-ACK arasında geçen süre (saniye)"],
    ["ackdat",  "ackdat",  "ACK-DATA arasında geçen süre (saniye)"],
    ["sjit",    "sjit",    "Kaynak tarafı jitter değeri (ms)"],
    ["djit",    "djit",    "Hedef tarafı jitter değeri (ms)"],
    ["ε",       "—",       "Sıfıra bölmeyi önlemek için eklenen küçük sabit (1e-10)"],
    ["Q_p(·)",  "—",       "p'inci persentil fonksiyonu"],
    ["d",       "—",       "Özellik vektörünün boyutu (toplam sütun sayısı)"],
    ["x_j",    "—",       "j-inci özelliğin değeri"],
    ["μ_j",    "—",       "j-inci özelliğin eğitim kümesi ortalaması"],
    ["σ_j",    "—",       "j-inci özelliğin eğitim kümesi standart sapması"],
]
add_table(doc, sym_headers, sym_rows, col_widths=[2.5, 3.0, 10.0])
doc.add_paragraph()


# ════════════════════════════════════════════════════════════════════════════
# GRUP A — İSTATİSTİKSEL ANOMALİ GÖSTERGELERİ
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "3. Grup A — İstatistiksel Anomali Göstergeleri (4 Öznitelik)", 1)
add_para(doc, (
    "Bu gruptaki öznitelikler, her örneğin (bağlantının) tüm ham özellik "
    "vektörü üzerinde satır bazlı hesaplanır. Yani tek bir bağlantıya ait "
    "tüm sayısal sütunlar birlikte değerlendirilerek o bağlantının genel "
    "anormallik düzeyi sayısal olarak ifade edilir. Bu yaklaşım, hangi "
    "özelliğin sapkın olduğundan bağımsız biçimde bütünsel sapkınlığı yakalar."
))

# A1
add_heading(doc, "A1. zscore_count — Z-Skor Aşım Sayısı", 2)
add_para(doc, (
    "Ham veri sütunları: Tüm sayısal özellik sütunları (d adet). "
    "Her sütun j için önce eğitim kümesinin ortalaması μ_j ve standart "
    "sapması σ_j hesaplanır. Ardından test örneği x'in j. sütundaki değeri "
    "bu parametrelerle standardize edilir ve mutlak z-skoru 3'ü aşan "
    "sütunlar sayılır."
))
add_formula(doc, "zscore_count(x) = Σ_{j=1}^{d}  1[ |( x_j - μ_j ) / σ_j| > 3 ]")
add_para(doc, "Semboller:")
add_bullet(doc, "x_j  : İncelenen bağlantının j. özelliğinin değeri")
add_bullet(doc, "μ_j  : j. özelliğin eğitim kümesi üzerinden hesaplanan aritmetik ortalaması")
add_bullet(doc, "σ_j  : j. özelliğin eğitim kümesi üzerinden hesaplanan standart sapması")
add_bullet(doc, "1[·] : İçindeki koşul doğruysa 1, yanlışsa 0 döndüren gösterge fonksiyonu")
add_bullet(doc, "d    : Ham özellik vektörünün toplam boyutu")
add_para(doc, (
    "Hesaplama: Preprocessing sonrası elde edilen Xt_tr matrisi üzerinde "
    "scipy.stats.zscore(X, axis=0) ile sütun bazlı z-skorlar hesaplanır. "
    "Ardından mutlak değerleri 3'ü aşan sütunlar axis=1 (satır bazlı) "
    "toplanarak her örnek için tek bir skalar üretilir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "DoS, Exploits, Fuzzers — yüksek hacimli veya bozuk paket gönderen saldırılar birden fazla sütunda aşırı değer üretir.")
add_bullet(doc, "Shellcode, Analysis — payload boyutu ve paket sayısı gibi kritik sütunlarda ani sapmalar görülür.")
add_para(doc, (
    "Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. "
    "ACM Computing Surveys, 41(3), 15:1–15:58. https://doi.org/10.1145/1541880.1541882"
), italic=True)

# A2
add_heading(doc, "A2. extreme_count — Uç Değer Sayısı (%1 / %99 Persentil)", 2)
add_para(doc, (
    "Ham veri sütunları: Tüm sayısal özellik sütunları (d adet). "
    "Her sütun j için eğitim kümesi üzerinden %1 persentil (p1_j) ve "
    "%99 persentil (p99_j) hesaplanır. Test örneğindeki her değer bu "
    "sınırların dışındaysa uç değer sayılır."
))
add_formula(doc, "extreme_count(x) = Σ_{j=1}^{d}  1[ x_j ≤ p1_j  OR  x_j ≥ p99_j ]")
add_para(doc, "Semboller:")
add_bullet(doc, "p1_j  : j. özelliğin eğitim kümesi %1 persentili (alt sınır)")
add_bullet(doc, "p99_j : j. özelliğin eğitim kümesi %99 persentili (üst sınır)")
add_bullet(doc, "OR    : Mantıksal veya; her iki koşuldan biri sağlandığında gösterge 1 döner")
add_para(doc, (
    "Hesaplama: numpy.percentile(Xt_tr, 1, axis=0) ve percentile(Xt_tr, 99, axis=0) "
    "ile her sütun için eşik değerleri belirlenir. Sonra (X <= p1) | (X >= p99) "
    "boolean matrisi oluşturulup .sum(axis=1) ile satır bazlı toplanır. "
    "Z-skora göre avantajı: Dağılımın şeklinden bağımsız, dağılıma uyumu "
    "gerektirmeyen (non-parametric) bir göstergedir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Genel anomali göstergesi — tüm saldırı sınıflarına katkı sağlar.")
add_bullet(doc, "Özellikle Worms ve Fuzzers — düzinelerce özellikte aynı anda sınır dışı değerler oluşturur.")
add_para(doc, (
    "Agyemang, M., Barker, K., & Alhajj, R. (2006). A Comprehensive Survey of Numeric "
    "and Symbolic Outlier Mining Techniques. Intelligent Data Analysis, 10(6), 521–538."
), italic=True)


# A3-A6
add_heading(doc, "A3–A6. İstatistiksel Momentler (Ortalama, Std, Çarpıklık, Basıklık)", 2)
add_para(doc, (
    "Ham veri sütunları: Tüm sayısal özellik sütunları. Her örnek (bağlantı) için "
    "d boyutlu özellik vektörü x üzerinde dört istatistiksel moment hesaplanır."
))
add_formula(doc, "mean(x)     = (1/d) * Σ_{j=1}^{d} x_j")
add_formula(doc, "std(x)      = sqrt( (1/d) * Σ_{j=1}^{d} (x_j - mean(x))^2 )")
add_formula(doc, "skewness(x) = [ (1/d) * Σ (x_j - mean)^3 ] / std^3")
add_formula(doc, "kurtosis(x) = [ (1/d) * Σ (x_j - mean)^4 ] / std^4  -  3")
add_para(doc, "Semboller:")
add_bullet(doc, "mean(x) : Özellik vektörünün aritmetik ortalaması — vektörün merkezi konumu")
add_bullet(doc, "std(x)  : Standart sapma — vektördeki değerlerin ortalamadan ortalama uzaklığı")
add_bullet(doc, "skewness: Üçüncü merkezi moment — dağılımın simetri eksikliğini ölçer; "
                "pozitif: sağ kuyruk ağır, negatif: sol kuyruk ağır")
add_bullet(doc, "kurtosis: Dördüncü merkezi moment eksi 3 (excess kurtosis) — "
                "dağılımın normal dağılıma göre sivrilik/yassılık farkını ölçer")
add_para(doc, (
    "Hesaplama: scipy.stats.skew(X, axis=1) ve scipy.stats.kurtosis(X, axis=1) "
    "ile her satır için hesaplanır. axis=1 parametresi hesabın her örnek (bağlantı) "
    "üzerinde, özellik boyutu boyunca yapıldığını belirtir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Yüksek çarpıklık: DoS, Fuzzers — anormal yüksek byte/paket değerleri dağılımı sağa çeker.")
add_bullet(doc, "Yüksek basıklık: Shellcode, Analysis — dar aralıkta yoğunlaşan ya da sivri uçlu dağılımlar.")
add_para(doc, (
    "Kira, R., & Rendell, L. A. (1992). A Practical Approach to Feature Selection. "
    "Proceedings of the 9th International Conference on Machine Learning (ICML), 249–256."
), italic=True)

# ════════════════════════════════════════════════════════════════════════════
# GRUP B — ANOMALİ SKORU
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4. Grup B — Anomali Skoru (1 Öznitelik)", 1)

add_heading(doc, "B1. isolation_forest_score — Isolation Forest Karar Skoru", 2)
add_para(doc, (
    "Ham veri sütunları: Preprocessing sonrası tüm sayısal özellik matrisi Xt_tr. "
    "Isolation Forest, her örneği diğerlerinden izole etmek için gereken ortalama "
    "yol uzunluğunu hesaplar. Kısa yol = kolay izolasyon = anormal örnek."
))
add_formula(doc, "s(x, n) = 2^( -E[h(x)] / c(n) )")
add_formula(doc, "c(n) = 2·H(n-1) - 2·(n-1)/n")
add_formula(doc, "H(i) = ln(i) + 0.5772...  (Euler-Mascheroni sabiti)")
add_para(doc, "Semboller:")
add_bullet(doc, "E[h(x)] : x örneğini izole etmek için gerekli ortalama yol uzunluğu (ağaç derinliği)")
add_bullet(doc, "c(n)    : n örnekli bir veri setinde ortalama yol uzunluğunun beklenen değeri (normalizasyon sabiti)")
add_bullet(doc, "n       : Eğitim kümesi büyüklüğü")
add_bullet(doc, "H(i)    : i'ye kadar harmonik sayı — ln(i) + Euler-Mascheroni sabiti (≈0.5772) ile yaklaşık")
add_bullet(doc, "s(x,n)  : Anomali skoru — 0.5'e yakın: normal, 1'e yakın: anormal")
add_para(doc, (
    "Hesaplama: sklearn.ensemble.IsolationForest(contamination=0.1, n_jobs=-1, "
    "random_state=42).fit(Xt_tr).decision_function(X). decision_function negatif "
    "değerler döndürür; daha negatif = daha anormal. contamination=0.1 parametresi "
    "eğitim verisinin %10'unun anormal olduğunu varsayar ve karar sınırını buna göre konumlandırır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Analysis, Shellcode — düşük örnekli, istatistiksel olarak nadir saldırı sınıfları.")
add_bullet(doc, "Worms — küçük veri kümesindeki örüntüler, kural tabanlı yaklaşımları atlar ama izolasyona yatkındır.")
add_para(doc, (
    "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. "
    "Proceedings of the 8th IEEE International Conference on Data Mining (ICDM), 413–422. "
    "https://doi.org/10.1109/ICDM.2008.17"
), italic=True)


# ════════════════════════════════════════════════════════════════════════════
# GRUP C — ALAN BİLGİSİ ÖZELLİKLERİ
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5. Grup C — Alan Bilgisi Özellikleri (18 Öznitelik)", 1)
add_para(doc, (
    "Bu gruptaki öznitelikler, her bir bağlantıya ait ham ağ akışı sütunlarından "
    "doğrudan türetilen siber güvenlik alan bilgisi içeren göstergelerdir. "
    "Her özellik belirli saldırı davranışlarını modellemiş akademik çalışmalara "
    "dayanmaktadır."
))

# C1
add_heading(doc, "C1. communication_efficiency — İletişim Verimliliği", 2)
add_para(doc, "Ham veri sütunları: sbytes, dbytes, spkts, dpkts")
add_para(doc, (
    "Her iki yönde iletilen toplam byte sayısını toplam paket sayısına böler. "
    "Paket başına düşen ortalama bilgi miktarını ölçer."
))
add_formula(doc, "CE = (sbytes + dbytes) / (spkts + dpkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sbytes : Kaynak→Hedef toplam iletilen byte (gönderilen payload)")
add_bullet(doc, "dbytes : Hedef→Kaynak toplam iletilen byte (alınan payload)")
add_bullet(doc, "spkts  : Kaynak→Hedef toplam gönderilen paket sayısı")
add_bullet(doc, "dpkts  : Hedef→Kaynak toplam gönderilen paket sayısı")
add_bullet(doc, "ε      : 1×10⁻¹⁰, sıfıra bölme koruması")
add_para(doc, (
    "Hesaplama: (df['sbytes']+df['dbytes']) / (df['spkts']+df['dpkts']+1e-10). "
    "Sonuç byte/paket birimiyle ifade edilir. Botnet C2 trafiği "
    "küçük ve sabit boyutlu kontrol mesajları içerdiğinden bu değer karakteristik "
    "dar bir aralıkta kalır; normal web trafiğinde ise yüksek varyans görülür."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Worms, Backdoor — C2 iletişimi sabit küçük paketler üretir.")
add_bullet(doc, "Generic — ortalama payload boyutundan sapan genel saldırılar.")
add_para(doc, (
    "Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014). An Empirical Comparison "
    "of Botnet Detection Methods. Computers & Security, 45, 100–123. "
    "https://doi.org/10.1016/j.cose.2014.05.011"
), italic=True)

# C2
add_heading(doc, "C2. activity_intensity — Akış Yoğunluğu", 2)
add_para(doc, "Ham veri sütunları: spkts, dpkts, dur")
add_para(doc, "Birim zamandaki toplam paket değişim sayısını ölçer.")
add_formula(doc, "AI = (spkts + dpkts) / (dur + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "spkts + dpkts : Her iki yönde gönderilen toplam paket — bağlantının toplam aktivitesi")
add_bullet(doc, "dur           : Bağlantı süresi saniye cinsinden")
add_para(doc, (
    "Hesaplama: (df['spkts']+df['dpkts']) / (df['dur']+1e-10). "
    "Saniyedeki toplam paket (paket/s) biriminde. DoS saldırıları bant genişliğini "
    "tüketmek için çok sayıda paketi kısa süreye sıkıştırır; tarama araçları ise "
    "paralel bağlantı kurarak saniyede çok sayıda küçük paket üretir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Reconnaissance, DoS, Fuzzers — yüksek AI değeri üretir.")
add_para(doc, (
    "Mirkovic, J., & Reiher, P. (2004). A Taxonomy of DDoS Attack and DDoS Defense "
    "Mechanisms. ACM SIGCOMM Computer Communication Review, 34(2), 39–53."
), italic=True)

# C3
add_heading(doc, "C3. is_long_duration — Uzun Süre Göstergesi", 2)
add_para(doc, "Ham veri sütunları: dur")
add_para(doc, (
    "Eğitim kümesindeki dur sütununun %90 persentil değeri eşik olarak belirlenir. "
    "Bağlantı bu eşiği aşıyorsa 1, geçmiyorsa 0 döner."
))
add_formula(doc, "is_long_duration = 1[ dur > Q_0.90(dur_train) ]")
add_para(doc, "Semboller:")
add_bullet(doc, "dur              : Mevcut bağlantının süresi (saniye)")
add_bullet(doc, "Q_0.90(dur_train): Eğitim kümesindeki dur değerlerinin 90. persentili")
add_bullet(doc, "1[·]             : Koşul doğruysa 1, yanlışsa 0")
add_para(doc, (
    "Hesaplama: df['dur'].quantile(0.9) ile eşik hesaplanır, ardından "
    "(df['dur'] > eşik).astype(float). Backdoor bağlantıları kalıcı ve "
    "gizli tünel kurar; bu da sürenin uzun olmasına neden olur."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Backdoor — kalıcı komuta kontrol tüneli uzun süreli bağlantı gerektirir.")
add_para(doc, (
    "Bhuyan, M. H., Bhattacharyya, D. K., & Kalita, J. K. (2014). Network Anomaly "
    "Detection: Methods, Systems and Tools. IEEE Communications Surveys & Tutorials, "
    "16(1), 303–336. https://doi.org/10.1109/SURV.2013.052213.00046"
), italic=True)


# C4
add_heading(doc, "C4. uses_non_standard_port — Standart Dışı Port Kullanımı", 2)
add_para(doc, "Ham veri sütunları: dsport")
add_para(doc, (
    "Hedef port numarasının bilinen standart servis portları listesinde "
    "olup olmadığını kontrol eder."
))
add_formula(doc, "NSP = 1[ dsport ∉ {21, 22, 25, 53, 80, 110, 143, 443, 993, 995} ]")
add_para(doc, "Semboller:")
add_bullet(doc, "dsport : Hedef port numarası (0–65535 tam sayı)")
add_bullet(doc, "∉      : 'Üyesi değil' — kümedeki portlardan hiçbiriyle eşleşmiyorsa 1 döner")
add_bullet(doc, "Küme   : FTP(21), SSH(22), SMTP(25), DNS(53), HTTP(80), POP3(110), IMAP(143), HTTPS(443), IMAPS(993), POP3S(995)")
add_para(doc, (
    "Hesaplama: (~df['dsport'].isin([21,22,25,53,80,110,143,443,993,995])).astype(float). "
    "Güvenlik duvarları standart portlara göre yapılandırılır; saldırganlar bu kuralları "
    "atlatmak için yüksek veya rastgele portlar seçer."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Backdoor, Shellcode — güvenlik duvarı atlatmak için standart dışı port kullanımı.")
add_para(doc, (
    "Shiravi, A., Shiravi, H., Tavallaee, M., & Ghorbani, A. A. (2012). Toward Developing "
    "a Systematic Approach to Generate Benchmark Datasets for Intrusion Detection. "
    "Computers & Security, 31(3), 357–374."
), italic=True)

# C5
add_heading(doc, "C5. packet_rate — Paket Gönderim Hızı", 2)
add_para(doc, "Ham veri sütunları: spkts, dur")
add_formula(doc, "PR = spkts / (dur + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "spkts : Kaynak→Hedef yönünde gönderilen paket sayısı")
add_bullet(doc, "dur   : Bağlantı süresi (saniye)")
add_para(doc, (
    "Hesaplama: df['spkts'] / (df['dur']+1e-10). Paket/saniye birimiyle ifade edilir. "
    "Flood tabanlı DoS saldırıları saniyede on binlerce paket üretir; "
    "normal bağlantılarda bu değer çok daha düşüktür."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "DoS — yüksek paket hızı temel DoS göstergesi.")
add_bullet(doc, "Fuzzers — çok sayıda kısa bağlantı benzer yüksek oranlar üretir.")
add_para(doc, (
    "Lippmann, R., Haines, J., Fried, D., Korba, J., & Das, K. (2000). The 1999 DARPA "
    "Off-Line Intrusion Detection Evaluation. Computer Networks, 34(4), 579–595."
), italic=True)

# C6
add_heading(doc, "C6. byte_rate — Byte Gönderim Hızı", 2)
add_para(doc, "Ham veri sütunları: sbytes, dur")
add_formula(doc, "BR = sbytes / (dur + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sbytes : Kaynak tarafından gönderilen toplam byte (payload dahil)")
add_bullet(doc, "dur    : Bağlantı süresi (saniye)")
add_para(doc, (
    "Hesaplama: df['sbytes'] / (df['dur']+1e-10). Byte/saniye birimiyle. "
    "Volumetrik DoS saldırıları bant genişliğini doyurmak için yüksek byte hızı üretir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "DoS, Generic — yüksek byte hızı flood saldırısı işareti.")
add_para(doc, (
    "Mirkovic, J., & Reiher, P. (2004). A Taxonomy of DDoS Attack and DDoS Defense "
    "Mechanisms. ACM SIGCOMM CCR, 34(2), 39–53."
), italic=True)

# C7
add_heading(doc, "C7. payload_asymmetry — Yük Asimetrisi", 2)
add_para(doc, "Ham veri sütunları: sbytes, dbytes")
add_para(doc, (
    "Kaynak ve hedef yük büyüklükleri arasındaki normalize edilmiş mutlak farkı ölçer."
))
add_formula(doc, "PA = |sbytes - dbytes| / (sbytes + dbytes + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "|·|    : Mutlak değer — farkın yönünden bağımsız büyüklüğünü alır")
add_bullet(doc, "Payda  : sbytes + dbytes, normalize etmek için toplam yük boyutu")
add_para(doc, (
    "Hesaplama: np.abs(df['sbytes']-df['dbytes']) / (df['sbytes']+df['dbytes']+1e-10). "
    "0'a yakın: simetrik (normal iletişim), 1'e yakın: tek yönlü yük aktarımı. "
    "Buffer overflow exploit'lerinde saldırgan küçük istek gönderir, büyük yanıt alır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Exploits — küçük istek, büyük yanıt asimetrisi karakteristik.")
add_para(doc, (
    "Brugger, S. T., & Chow, J. (2007). An Assessment of the DARPA IDS Evaluation "
    "Dataset Using Snort. UCDAVIS Technical Report CSE-2007-1."
), italic=True)

# C8
add_heading(doc, "C8. packet_asymmetry — Paket Sayısı Asimetrisi", 2)
add_para(doc, "Ham veri sütunları: spkts, dpkts")
add_formula(doc, "PAK = |spkts - dpkts| / (spkts + dpkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "spkts  : Kaynak→Hedef paket sayısı")
add_bullet(doc, "dpkts  : Hedef→Kaynak paket sayısı")
add_bullet(doc, "Payda  : Toplam paket sayısı — normalize etmek için kullanılır")
add_para(doc, (
    "Hesaplama: np.abs(df['spkts']-df['dpkts']) / (df['spkts']+df['dpkts']+1e-10). "
    "SYN flood saldırısında hedeften yanıt gelmediğinden spkts >> dpkts olur. "
    "Yarım açık (half-open) bağlantı saldırıları da yüksek asimetri üretir."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Exploits, DoS — tek yönlü paket akışı belirgin asimetri üretir.")
add_para(doc, (
    "Chen, Y., Hwang, K., & Ku, W. S. (2007). Collaborative Detection of DDoS Attacks "
    "over Multiple Network Domains. IEEE Transactions on Parallel and Distributed "
    "Systems, 18(12), 1649–1662."
), italic=True)


# C9
add_heading(doc, "C9. avg_packet_size — Ortalama Paket Boyutu", 2)
add_para(doc, "Ham veri sütunları: sbytes, spkts")
add_formula(doc, "APS = sbytes / (spkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sbytes : Kaynak tarafından gönderilen toplam byte")
add_bullet(doc, "spkts  : Kaynak tarafından gönderilen toplam paket sayısı")
add_para(doc, (
    "Hesaplama: df['sbytes'] / (df['spkts']+1e-10). Byte/paket birimi. "
    "Fuzzer araçları protokol sınırlarını test etmek için çok sayıda küçük "
    "ve bozuk paket gönderir — ortalama paket boyutu düşük çıkar. "
    "Normal HTTP trafiğinde bu değer 500–1500 byte arasındadır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Fuzzers — küçük test paketleri düşük APS üretir.")
add_bullet(doc, "Reconnaissance — tarama paketleri de küçük boyutludur.")
add_para(doc, (
    "Amalfitano, D., Fasolino, A. R., & Tramontana, P. (2012). A General Framework "
    "for Fuzzing. IEEE Transactions on Software Engineering, 38(6), 1248–1266."
), italic=True)

# C10
add_heading(doc, "C10. is_very_short_duration — Çok Kısa Süre Göstergesi", 2)
add_para(doc, "Ham veri sütunları: dur")
add_formula(doc, "VSD = 1[ dur ≤ Q_0.10(dur_train) ]")
add_para(doc, "Semboller:")
add_bullet(doc, "dur              : Bağlantı süresi (saniye)")
add_bullet(doc, "Q_0.10(dur_train): Eğitim kümesi dur değerlerinin 10. persentili")
add_para(doc, (
    "Hesaplama: eşik = df['dur'].quantile(0.1), ardından "
    "(df['dur'] <= eşik).astype(float). Fuzzer'lar bağlantıyı çok hızlı açıp "
    "kapattığından süre son derece düşük olur. %10 persentil eşiği, veri setine "
    "göre otomatik ayarlanır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Fuzzers — anlık test bağlantıları.")
add_bullet(doc, "Reconnaissance — hızlı port tarama bağlantıları.")
add_para(doc, (
    "Sutton, M., Greene, A., & Amini, P. (2007). Fuzzing: Brute Force Vulnerability "
    "Discovery. Addison-Wesley Professional."
), italic=True)

# C11
add_heading(doc, "C11. src_port_entropy — Kaynak Port Çeşitliliği", 2)
add_para(doc, "Ham veri sütunları: sport, srcip")
add_para(doc, (
    "Aynı kaynak IP'den gelen bağlantılarda kullanılan benzersiz kaynak port "
    "sayısının toplam bağlantı sayısına oranı. Kaynak IP başına hesaplanır, "
    "sonra her satıra yayılır."
))
add_formula(doc, "SPE(s) = |{ sport_i : src_i = s }| / |{ i : src_i = s }| + ε")
add_para(doc, "Semboller:")
add_bullet(doc, "s          : Kaynak IP adresi")
add_bullet(doc, "|{sport_i}|: s'den gelen bağlantılarda kullanılan benzersiz kaynak port sayısı")
add_bullet(doc, "|{i}|      : s'den gelen toplam bağlantı sayısı")
add_para(doc, (
    "Hesaplama: df.groupby('srcip')['sport'].transform('nunique') / "
    "(df.groupby('srcip')['sport'].transform('count') + 1e-10). "
    "Port tarama araçları her hedef porta bağlanırken rastgele kaynak port seçer — "
    "benzersiz kaynak port oranı 1'e yakın olur. Normal istemciler az sayıda "
    "kaynak port kullanır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Reconnaissance — rastgele kaynak port kullanan tarama araçları.")
add_para(doc, (
    "Nychis, G., Sekar, V., Andersen, D. G., Kim, H., & Zhang, H. (2008). "
    "An Empirical Evaluation of Entropy-based Traffic Anomaly Detection. "
    "Proceedings of ACM IMC, 151–156."
), italic=True)

# C12
add_heading(doc, "C12. scans_common_port — Yaygın Tarama Portu Kullanımı", 2)
add_para(doc, "Ham veri sütunları: dsport")
add_formula(doc, "SCP = 1[ dsport ∈ {21,22,23,25,53,80,110,111,135,139,143,443,993,995,1433,3389} ]")
add_para(doc, "Semboller:")
add_bullet(doc, "dsport : Hedef port numarası")
add_bullet(doc, "Küme   : FTP(21),SSH(22),Telnet(23),SMTP(25),DNS(53),HTTP(80),POP3(110),")
add_bullet(doc, "         RPC(111),DCOM(135),NetBIOS(139),IMAP(143),HTTPS(443),")
add_bullet(doc, "         IMAPS(993),POP3S(995),MSSQL(1433),RDP(3389)")
add_para(doc, (
    "Hesaplama: df['dsport'].isin([21,22,...,3389]).astype(float). "
    "Bu 16 port, ağ tarama araçlarının (nmap, masscan) varsayılan hedefleridir. "
    "Aktif keşif saldırıları bu portları sistematik biçimde tarar."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Reconnaissance — yaygın servis portlarına yönelik tarama.")
add_para(doc, (
    "Paxson, V. (1999). Bro: A System for Detecting Network Intruders in Real-Time. "
    "Computer Networks, 31(23–24), 2435–2463."
), italic=True)

# C13
add_heading(doc, "C13. uses_small_packets — Küçük Paket Kullanım Göstergesi", 2)
add_para(doc, "Ham veri sütunları: sbytes, spkts")
add_formula(doc, "USP = 1[ sbytes / (spkts + ε) < 100 ]")
add_para(doc, "Semboller:")
add_bullet(doc, "Eşik 100 byte: Tarama paketlerinin tipik maksimum yük boyutu; ICMP, TCP SYN bu sınırın altında kalır")
add_para(doc, (
    "Hesaplama: (df['sbytes'] / (df['spkts']+1e-10) < 100).astype(float). "
    "Ping taraması (ICMP echo) ve TCP SYN taraması 40–80 byte civarı paketler kullanır. "
    "100 byte eşiği bu paketleri normal HTTP/HTTPS paketlerinden ayırır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Reconnaissance — minimum boyutlu tarama paketleri.")
add_para(doc, (
    "Staniford, S., Moore, D., Paxson, V., & Weaver, N. (2002). Practical Automated "
    "Detection of Stealthy Portscans. Journal of Computer Security, 10(1–2), 105–136."
), italic=True)


# C14
add_heading(doc, "C14. payload_density — Yük Yoğunluğu", 2)
add_para(doc, "Ham veri sütunları: sbytes, spkts")
add_formula(doc, "PD = sbytes / (spkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sbytes : Kaynak→Hedef toplam byte (shellcode yükü dahil)")
add_bullet(doc, "spkts  : Kaynak→Hedef toplam paket sayısı")
add_para(doc, (
    "Hesaplama: df['sbytes'] / (df['spkts']+1e-10). avg_packet_size ile "
    "aynı formüldür, ancak shellcode bağlamında yorumlanır: shellcode genellikle "
    "tek bir TCP segmentine sığacak kadar küçük ama yeterince büyük bir yük (100–2000 byte) "
    "içerir. Bu yük yoğunluğu değeri normal trafikten sistematik biçimde farklıdır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Shellcode — karakteristik yük boyutu aralığı.")
add_para(doc, (
    "Polychronakis, M., Anagnostakis, K. G., & Markatos, E. P. (2011). "
    "Network-level Polymorphic Shellcode Detection Using Emulation. "
    "Journal of Computer Virology, 7(4), 257–274."
), italic=True)

# C15
add_heading(doc, "C15. shellcode_size_range — Shellcode Boyut Aralığı Göstergesi", 2)
add_para(doc, "Ham veri sütunları: sbytes")
add_formula(doc, "SSR = 1[ 100 ≤ sbytes ≤ 2000 ]")
add_para(doc, "Semboller:")
add_bullet(doc, "100 byte alt sınır : İşlevsel shellcode'un taşıması gereken minimum sistem çağrısı yükü")
add_bullet(doc, "2000 byte üst sınır: Standart TCP MSS (Maximum Segment Size) yaklaşık 1460 byte — shellcode tek segment içinde kalır")
add_para(doc, (
    "Hesaplama: ((df['sbytes']>=100) & (df['sbytes']<=2000)).astype(float). "
    "Bu aralık işletim sistemi çağrısı yapabilen en küçük shellcode ile "
    "standart Ethernet segmentine sığan en büyük payload arasını kapsar. "
    "NOP sled içeren klasik buffer overflow istismarları bu aralığa girer."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Shellcode — işletim sistemi çağrısı yapan minimum yük boyutu.")
add_para(doc, (
    "Polychronakis, M., Anagnostakis, K. G., & Markatos, E. P. (2007). "
    "Emulation-based Detection of Non-self-contained Polymorphic Shellcode. "
    "Proceedings of RAID 2007, LNCS 4637, 87–106."
), italic=True)

# C16
add_heading(doc, "C16. uses_high_port — Yüksek Port Kullanım Göstergesi", 2)
add_para(doc, "Ham veri sütunları: dsport")
add_formula(doc, "UHP = 1[ dsport > 1024 ]")
add_para(doc, "Semboller:")
add_bullet(doc, "1024 : IANA'nın 'well-known ports' (0–1023) ile 'registered/dynamic ports' (1024–65535) arasındaki sınır")
add_para(doc, (
    "Hesaplama: (df['dsport'] > 1024).astype(float). "
    "Güvenlik duvarı politikaları genellikle 0–1023 arasındaki portlara katı "
    "kurallar uygular; 1024 üstü portlar daha az denetlenir. Shellcode ve backdoor "
    "programları bu nedenle dinamik port aralığını hedefler."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Shellcode, Backdoor — firewall atlatmak için yüksek port kullanımı.")
add_para(doc, (
    "IANA (2022). Service Name and Transport Protocol Port Number Registry. "
    "https://www.iana.org/assignments/service-names-port-numbers"
), italic=True)

# C17
add_heading(doc, "C17. moderate_throughput — Orta Seviye Verim Göstergesi", 2)
add_para(doc, "Ham veri sütunları: sbytes, dur")
add_formula(doc, "T(x) = sbytes / (dur + ε)")
add_formula(doc, "MT = 1[ Q_0.25(T) < T(x) < Q_0.75(T) ]")
add_para(doc, "Semboller:")
add_bullet(doc, "T(x)       : İncelenen bağlantının byte/saniye cinsinden verimi")
add_bullet(doc, "Q_0.25(T)  : Eğitim kümesi verimleri dağılımının 1. çeyreği")
add_bullet(doc, "Q_0.75(T)  : Eğitim kümesi verimleri dağılımının 3. çeyreği (IQR sınırları)")
add_para(doc, (
    "Hesaplama: throughput = df['sbytes']/(df['dur']+1e-10), ardından "
    "((throughput > np.percentile(throughput,25)) & (throughput < np.percentile(throughput,75))).astype(float). "
    "Generic saldırılar ne çok yüksek (DoS gibi) ne çok düşük (tarama gibi) verim üretir; "
    "IQR (çeyrekler arası aralık) içinde yoğunlaşır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Generic — orta seviye aktivite gösteren genel saldırı trafiği.")
add_para(doc, (
    "Moustafa, N., & Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set for "
    "Network Intrusion Detection Systems. Proceedings of MilCIS 2015, IEEE."
), italic=True)


# ════════════════════════════════════════════════════════════════════════════
# GRUP D — DERİN SİBER GÜVENLİK ÖZNİTELİKLERİ
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "6. Grup D — Derin Siber Güvenlik Özellikleri (7 Öznitelik)", 1)
add_para(doc, (
    "Bu gruptaki öznitelikler, TCP/IP protokol yığınının daha alt katmanlarına "
    "ait istatistikleri (kayıp oranı, TTL, jitter, TCP zamanlama) kullanarak "
    "ağ davranışının nüanslı özelliklerini yakalar."
))

# D1
add_heading(doc, "D1. s_loss_ratio — Kaynak Paket Kayıp Oranı", 2)
add_para(doc, "Ham veri sütunları: sloss, spkts")
add_formula(doc, "SLR = sloss / (spkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sloss : Kaynak→Hedef yönünde iletim sırasında tespit edilen kayıp paket sayısı")
add_bullet(doc, "spkts : Kaynak→Hedef gönderilen toplam paket sayısı")
add_para(doc, (
    "Hesaplama: df['sloss'] / (df['spkts']+1e-10). "
    "Fuzzer araçları bilerek bozuk veya eksik paketler gönderir; "
    "Analysis saldırıları protokol davranışını test etmek için kayıpları tetikler. "
    "0'a yakın: normal iletim, 1'e yakın: yüksek kayıplı/bozuk trafik."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Analysis, Fuzzers — kasıtlı bozuk paket üretimi yüksek kayıp oranı oluşturur.")
add_para(doc, (
    "Moustafa, N., & Slay, J. (2016). The Evaluation of Network Anomaly Detection "
    "Systems: Statistical Analysis of the UNSW-NB15 Data Set and the Comparison with "
    "the KDD99 Data Set. Security and Communication Networks, 9(15), 2099–2102."
), italic=True)

# D2
add_heading(doc, "D2. d_loss_ratio — Hedef Paket Kayıp Oranı", 2)
add_para(doc, "Ham veri sütunları: dloss, dpkts")
add_formula(doc, "DLR = dloss / (dpkts + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "dloss : Hedef→Kaynak yönünde tespit edilen kayıp paket sayısı")
add_bullet(doc, "dpkts : Hedef→Kaynak gönderilen toplam paket sayısı")
add_para(doc, (
    "Hesaplama: df['dloss'] / (df['dpkts']+1e-10). Hedefin saldırıya yanıt "
    "verirken düşürdüğü paketler ölçülür. Hedefe ulaşmayan ya da "
    "kaynak tarafından işlenemeyen yanıt paketleri bu orana yansır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Analysis, Fuzzers — protokol test trafiğinde hedef tarafında da kayıplar oluşur.")
add_para(doc, "Moustafa, N., & Slay, J. (2016). (Yukarıda anılan aynı kaynak.)", italic=True)

# D3
add_heading(doc, "D3. loss_asymmetry — Paket Kayıp Asimetrisi", 2)
add_para(doc, "Ham veri sütunları: sloss, spkts, dloss, dpkts")
add_formula(doc, "LA = | SLR - DLR |  =  | sloss/(spkts+ε)  -  dloss/(dpkts+ε) |")
add_para(doc, "Semboller:")
add_bullet(doc, "SLR   : Kaynak paket kayıp oranı (D1)")
add_bullet(doc, "DLR   : Hedef paket kayıp oranı (D2)")
add_bullet(doc, "|·|   : Mutlak değer — asimetrinin yönünden bağımsız büyüklüğü")
add_para(doc, (
    "Hesaplama: np.abs(s_loss_ratio - d_loss_ratio). "
    "IP spoofing'de kaynak adresi sahte olduğu için kaynak tarafı istatistikleri "
    "ile hedef tarafı istatistikleri uyumsuz olur — büyük asimetri oluşur. "
    "Normal çift yönlü TCP bağlantılarında kayıp oranları yakın değerler alır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Analysis, Exploits — spoofed kaynak ile bağlantı asimetrisi.")
add_para(doc, (
    "Handley, M., Paxson, V., & Kreibich, C. (2001). Network Intrusion Detection: "
    "Evasion, Traffic Normalization, and End-to-End Protocol Semantics. "
    "Proceedings of USENIX Security Symposium, 115–131."
), italic=True)

# D4
add_heading(doc, "D4. ttl_asymmetry — TTL Asimetrisi", 2)
add_para(doc, "Ham veri sütunları: sttl, dttl")
add_formula(doc, "TTLA = |sttl - dttl| / (sttl + dttl + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sttl   : Kaynak paketlerinde gözlemlenen TTL değeri")
add_bullet(doc, "dttl   : Hedef (yanıt) paketlerinde gözlemlenen TTL değeri")
add_bullet(doc, "Payda  : sttl+dttl, normalize için kullanılır; 0'a bölmeyi önler")
add_para(doc, (
    "Hesaplama: np.abs(df['sttl']-df['dttl']) / (df['sttl']+df['dttl']+1e-10). "
    "Normal TCP bağlantılarında her iki tarafın TTL değerleri bilinen işletim "
    "sistemi başlangıç değerlerine (64, 128, 255) göre hesaplanabilir aralıkta olur. "
    "IP spoofing'de saldırgan gerçek TTL'yi taklit edemez; tünel (backdoor) "
    "trafiğinde de gizlenmiş iç paketlerin TTL'si dış paketlerden farklıdır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Backdoor, Exploits, Shellcode — IP spoofing ve tünel trafiği belirgin TTL asimetrisi üretir.")
add_para(doc, (
    "Templeton, S. J., & Levitt, K. (2000). A Requires/Provides Model for Computer "
    "Attacks. Proceedings of the New Security Paradigms Workshop (NSPW), 31–38."
), italic=True)

# D5
add_heading(doc, "D5. tcp_setup_inefficiency — TCP Kurulum Verimsizliği", 2)
add_para(doc, "Ham veri sütunları: tcprtt, dur")
add_formula(doc, "TSI = tcprtt / (dur + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "tcprtt : TCP bağlantı kurulum aşamasının round-trip süresi (saniye)")
add_bullet(doc, "dur    : Toplam bağlantı süresi (saniye)")
add_para(doc, (
    "Hesaplama: df['tcprtt'] / (df['dur']+1e-10). "
    "Normal bağlantılarda TCP kurulum süresi toplam sürenin küçük bir payıdır. "
    "Slowloris saldırısında bağlantı kurulduktan sonra çok yavaş veri gönderilir — "
    "tcprtt/dur oranı anormal büyür. Fuzzer'larda bağlantı hemen kapandığından "
    "tcprtt ≈ dur olur ve oran 1'e yaklaşır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Fuzzers — tcprtt ≈ dur (bağlantı kurulur kurulmaz kapanır).")
add_bullet(doc, "Reconnaissance — hızlı bağlantı açıp kapama davranışı.")
add_para(doc, (
    "Roesch, M. (1999). Snort — Lightweight Intrusion Detection for Networks. "
    "Proceedings of LISA'99, 229–238."
), italic=True)

# D6
add_heading(doc, "D6. tcp_ack_syn_ratio — ACK/SYN Zaman Oranı", 2)
add_para(doc, "Ham veri sütunları: ackdat, synack")
add_formula(doc, "TASR = ackdat / (synack + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "ackdat : ACK paketi ile DATA (veri) transferi arasında geçen süre (saniye)")
add_bullet(doc, "synack : SYN gönderimi ile SYN-ACK alımı arasında geçen süre (saniye)")
add_para(doc, (
    "Hesaplama: df['ackdat'] / (df['synack']+1e-10). "
    "Normal TCP 3-way handshake'te synack ve ackdat değerleri ağ gecikmesininin "
    "katlarıdır — oran makul bir aralıkta kalır. SYN flood saldırılarında "
    "ackdat=0 (veri hiç gelmez) veya çok büyük olur, oranı bozar."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "Exploits, DoS — bozuk TCP el sıkışma zaman dizisi.")
add_para(doc, (
    "Schechter, S., Jung, J., & Berger, A. W. (2004). Fast Detection of Scanning "
    "Worm Infections. Proceedings of RAID 2004, LNCS 3224, 59–81."
), italic=True)

# D7
add_heading(doc, "D7. jitter_asymmetry — Jitter Asimetrisi", 2)
add_para(doc, "Ham veri sütunları: sjit, djit")
add_formula(doc, "JA = |sjit - djit| / (sjit + djit + ε)")
add_para(doc, "Semboller:")
add_bullet(doc, "sjit   : Kaynak paketleri arası varış zamanı varyasyonu — jitter (ms)")
add_bullet(doc, "djit   : Hedef (yanıt) paketleri arası varış zamanı varyasyonu — jitter (ms)")
add_bullet(doc, "Payda  : Toplam jitter, normalize etmek için")
add_para(doc, (
    "Hesaplama: np.abs(df['sjit']-df['djit']) / (df['sjit']+df['djit']+1e-10). "
    "DoS flood saldırıları ağ kuyruklarını doldurarak asimetrik gecikme "
    "dalgalanması oluşturur: saldırgan tarafı düzenli gönderir (düşük sjit), "
    "hedef tarafı aşırı yükten dolayı yüksek djit üretir veya tam tersi. "
    "Normal iletişimde her iki jitter değeri birbirine yakındır."
))
add_para(doc, "Katkı sağladığı saldırı sınıfları:", bold_prefix="Katkı: ")
add_bullet(doc, "DoS — ağ kuyruğu baskısından kaynaklanan asimetrik jitter.")
add_bullet(doc, "Fuzzers — düzensiz paket gönderimi jitter farklılığı yaratır.")
add_para(doc, (
    "Papagiannaki, K., Taft, N., Zhang, Z. L., & Diot, C. (2004). Long-Term Forecasting "
    "of Internet Backbone Traffic: Observations and Initial Models. "
    "IEEE Transactions on Network and Service Management, 1(1), 2–14."
), italic=True)


# ════════════════════════════════════════════════════════════════════════════
# ÖZET TABLOSU
# ════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
add_heading(doc, "7. Özet Tablosu — Tüm Öznitelikler", 1)
add_para(doc, (
    "Aşağıdaki tablo, mühendislik edilen tüm öznitelikleri grup, formül özeti, "
    "kullanılan ham sütunlar ve birincil saldırı sınıfı bilgileriyle birlikte özetlemektedir."
))

tbl_headers = ["#", "Öznitelik Adı", "Grup", "Kullanılan Ham Sütunlar",
               "Formül Özeti", "Birincil Saldırı Sınıfı"]
tbl_rows = [
    ["A1", "zscore_count",            "A", "Tüm sayısal sütunlar",      "Σ 1[|z_j|>3]",                     "DoS, Exploits, Fuzzers"],
    ["A2", "extreme_count",           "A", "Tüm sayısal sütunlar",      "Σ 1[x≤p1 OR x≥p99]",               "Genel anomali"],
    ["A3", "mean",                    "A", "Tüm sayısal sütunlar",      "Σx_j / d",                         "Genel"],
    ["A4", "std",                     "A", "Tüm sayısal sütunlar",      "sqrt(Σ(x-μ)²/d)",                  "Genel"],
    ["A5", "skewness",                "A", "Tüm sayısal sütunlar",      "Σ(x-μ)³ / (d·σ³)",                 "DoS, Fuzzers"],
    ["A6", "kurtosis",                "A", "Tüm sayısal sütunlar",      "Σ(x-μ)⁴/(d·σ⁴) - 3",              "DoS, Shellcode"],
    ["B1", "isolation_forest_score",  "B", "Tüm sayısal sütunlar",      "2^(-E[h(x)]/c(n))",                "Analysis, Shellcode"],
    ["C1", "communication_efficiency","C", "sbytes,dbytes,spkts,dpkts", "(sb+db)/(sp+dp+ε)",                "Worms, Backdoor"],
    ["C2", "activity_intensity",      "C", "spkts,dpkts,dur",           "(sp+dp)/(dur+ε)",                  "Recon, DoS, Fuzzers"],
    ["C3", "is_long_duration",        "C", "dur",                       "1[dur > Q90]",                     "Backdoor"],
    ["C4", "uses_non_standard_port",  "C", "dsport",                    "1[dsport ∉ standart]",              "Backdoor, Shellcode"],
    ["C5", "packet_rate",             "C", "spkts, dur",                "spkts/(dur+ε)",                    "DoS"],
    ["C6", "byte_rate",               "C", "sbytes, dur",               "sbytes/(dur+ε)",                   "DoS, Generic"],
    ["C7", "payload_asymmetry",       "C", "sbytes, dbytes",            "|sb-db|/(sb+db+ε)",                "Exploits"],
    ["C8", "packet_asymmetry",        "C", "spkts, dpkts",              "|sp-dp|/(sp+dp+ε)",                "Exploits, DoS"],
    ["C9", "avg_packet_size",         "C", "sbytes, spkts",             "sbytes/(spkts+ε)",                 "Fuzzers, Recon"],
    ["C10","is_very_short_duration",  "C", "dur",                       "1[dur ≤ Q10]",                     "Fuzzers"],
    ["C11","src_port_entropy",        "C", "sport, srcip",              "|unique sport|/|count|",           "Reconnaissance"],
    ["C12","scans_common_port",       "C", "dsport",                    "1[dsport ∈ {16 port}]",            "Reconnaissance"],
    ["C13","uses_small_packets",      "C", "sbytes, spkts",             "1[sb/sp < 100]",                   "Reconnaissance"],
    ["C14","payload_density",         "C", "sbytes, spkts",             "sbytes/(spkts+ε)",                 "Shellcode"],
    ["C15","shellcode_size_range",    "C", "sbytes",                    "1[100 ≤ sbytes ≤ 2000]",           "Shellcode"],
    ["C16","uses_high_port",          "C", "dsport",                    "1[dsport > 1024]",                 "Shellcode, Backdoor"],
    ["C17","moderate_throughput",     "C", "sbytes, dur",               "1[Q25<T<Q75]",                     "Generic"],
    ["D1", "s_loss_ratio",            "D", "sloss, spkts",              "sloss/(spkts+ε)",                  "Analysis, Fuzzers"],
    ["D2", "d_loss_ratio",            "D", "dloss, dpkts",              "dloss/(dpkts+ε)",                  "Analysis, Fuzzers"],
    ["D3", "loss_asymmetry",          "D", "sloss,spkts,dloss,dpkts",   "|SLR - DLR|",                      "Analysis, Exploits"],
    ["D4", "ttl_asymmetry",           "D", "sttl, dttl",                "|sttl-dttl|/(sttl+dttl+ε)",        "Backdoor, Exploits"],
    ["D5", "tcp_setup_inefficiency",  "D", "tcprtt, dur",               "tcprtt/(dur+ε)",                   "Fuzzers, Recon"],
    ["D6", "tcp_ack_syn_ratio",       "D", "ackdat, synack",            "ackdat/(synack+ε)",                "Exploits, DoS"],
    ["D7", "jitter_asymmetry",        "D", "sjit, djit",                "|sjit-djit|/(sjit+djit+ε)",        "DoS, Fuzzers"],
]
add_table(doc, tbl_headers, tbl_rows, col_widths=[0.7, 3.5, 0.8, 3.5, 3.5, 3.5])

# ════════════════════════════════════════════════════════════════════════════
# KAYDET
# ════════════════════════════════════════════════════════════════════════════
out_path = "oznitelik_analizi.docx"
doc.save(out_path)
print(f"Word dosyasi olusturuldu: {out_path}")
