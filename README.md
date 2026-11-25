# tr-duzelt

Türkçe metin düzeltme: **yazım, noktalama, büyük harf, kesme işareti**.
Ana düzeltme LLM kullanmadan klasik NLP (Zemberek morfolojisi + 2-gram dil modeli) ve küçük bir CRF modeliyle çalışır; yalnızca CPU ister, token harcamaz. İsteğe bağlı `--eval` son çıktıyı Gemini'ye gönderip şüpheli değişiklikleri işaretler; düzeltme metnini değiştirmez.

Bu bir prototiptir: neyi ne kadar yapabildiği ve **yapamadığı** aşağıda ölçümleriyle yazıyor.

## Örnekler

Çıktılar bu depodaki kod ve modelle üretilmiştir.

| Girdi | Çıktı |
|---|---|
| bugün okula gitmedim çünki hastaydım annem beni doktora götürdü doktor bişey yok dedi ama ben hala kötü hissediyorum yarın sınavım var napcam bilmiyorum ankarada hava çok soğuk | Bugün okula gitmedim çünkü hastaydım. Annem beni doktora götürdü. Doktor bir şey yok dedi. Ama ben hala kötü hissediyorum. Yarın sınavım var ne yapacağım bilmiyorum. Ankara'da hava çok soğuk. |
| kardeşim 3 mayısta doğdu o zamanlar izmirde yaşıyorduk babam bir fabrikada çalışıyordu | Kardeşim 3 Mayıs'ta doğdu. O zamanlar İzmir'de yaşıyorduk. Babam bir fabrikada çalışıyordu. |
| herkez 1994de doğmuş gibi davranıyo yanlız ben 2000li yılların çocuğuyum | Herkes 1994'te doğmuş gibi davranıyor. Yalnız ben 2000'li yılların çocuğuyum. |
| dun aksam arkadaslarimla sinemaya gittik film cok guzeldi sonra yemek yedik eve gec geldim | Dün akşam arkadaşlarımla sinemaya gittik. Film çok güzeldi. Sonra yemek yedik. Eve geç geldim. |
| dün arkadaşım ahmetle buluştuk o bana yeni telefonunu gösterdi çok beğendim bende almak istiyorum ama param yok | Dün arkadaşım Ahmet'le buluştuk. O bana yeni telefonunu gösterdi. Çok beğendim bende almak istiyorum ama param yok. |

Son satır bilerek burada: `bende` → `ben de` düzeltilmedi ve "beğendim"den sonra nokta konmadı. Nedeni [Sınırlar](#sınırlar) bölümünde.

## Kurulum

Python 3.11 ve 3.12 ile denendi. `zemberek-python`, Python 3.13+ ve NumPy 2 ile çalışmıyor.

```bash
pip install -r requirements.txt
```

### Windows (yönetici hakkı olmadan)

Bu yol hiçbir adımda yönetici onayı istemez: her şey kullanıcı klasörünüze kurulur, sistemdeki başka bir Python'a dokunulmaz ve C derleyicisi gerekmez.

> 3. ve 4. adımlar aynı uv komutlarıyla Linux'ta doğrulandı; gerçek bir Windows makinede henüz denenmedi. Takıldığınız yer olursa issue açın.

**1. Depoyu indirin.** [ZIP olarak indirip](https://github.com/oguzhanozfe/tr-duzelt/archive/refs/heads/main.zip) bir klasöre açın. Git gerekmez.

**2. uv'yi kurun.** [uv](https://docs.astral.sh/uv/), doğru Python sürümünü kendisi indirip yöneten bir araçtır. Başlat menüsünden PowerShell'i normal şekilde açın ("Yönetici olarak çalıştır" değil) ve şunu çalıştırın:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Bitince pencereyi kapatıp yeni bir PowerShell açın; `uv --version` sürüm yazmalı.

**3. Ortamı kurun.** Zip'i açtığınız, içinde `tr_duzelt.py` bulunan klasöre girin:

```powershell
cd $HOME\Downloads\tr-duzelt-main      # klasör sizde farklıysa onu yazın
uv venv --python 3.11
uv pip install -r requirements.txt
```

`uv venv`, bilgisayarda Python 3.11 yoksa onu kendisi indirir. İkinci komut yaklaşık 115 MB indirir (çoğu Zemberek'in dil modeli).

**4. Çalıştırın.**

```powershell
.venv\Scripts\python tr_duzelt.py "herkez geldimi bilmiyorum"
.venv\Scripts\python tr_duzelt.py --dosya girdi.txt --cikti cikti.txt
```

Ortamı etkinleştirmeniz (`activate`) gerekmez; `.venv\Scripts\python` doğrudan çalışır, böylece PowerShell'in betik çalıştırma kısıtına takılmazsınız. Uzun metinlerde `--dosya` / `--cikti` kullanın: çıktı UTF-8 yazılır, girdi UTF-8, UTF-16 ya da ANSI olabilir.

**Takılırsanız**

| Belirti | Çözüm |
|---|---|
| 2. adımdaki betik engelleniyor ya da `uv` bulunamıyor | [uv-x86_64-pc-windows-msvc.zip](https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip) dosyasını indirin, içindeki `uv.exe`'yi proje klasörüne koyun ve komutlarda `uv` yerine `.\uv.exe` yazın. Kurulum gerektirmez. |
| İndirmeler sertifika (SSL) hatası veriyor; kurumsal ağlarda olur | uv komutlarına `--system-certs` ekleyin, ör. `uv pip install --system-certs -r requirements.txt` (eski uv sürümlerinde `--native-tls`). |
| `ImportError: DLL load failed while importing _pycrfsuite` | Bilgisayarda Visual C++ çalışma zamanı (`MSVCP140.dll`) yok. Yönetici gerektirmeyen çözüm: `uv pip install msvc-runtime`; betik bu DLL'leri ortamın içinde arar. Yine de yüklenmezse `.venv\Scripts\msvcp140.dll` dosyasını `.venv\Lib\site-packages\pycrfsuite\` klasörüne kopyalayın. |
| uv kullanamıyorsanız | [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) sayfasından "Windows installer (64-bit)" indirin (3.11'in kurucusu olan son sürümü budur). İlk ekranın altındaki **"Use admin privileges when installing py.exe"** kutusunun işaretini kaldırıp *Install Now* deyin. Sonra proje klasöründe: `& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv` ve `.venv\Scripts\python -m pip install -r requirements.txt`. |

## Kullanım

```bash
python tr_duzelt.py "herkez geldimi bilmiyorum"
# Herkes geldi mi bilmiyorum.

python tr_duzelt.py --dosya girdi.txt --ayrinti            # yapılan kelime düzeltmelerini de listeler
python tr_duzelt.py --dosya girdi.txt --cikti cikti.txt    # sonucu UTF-8 dosyaya yazar

# PowerShell'de yalnızca bu oturum için API anahtarı tanımla, sonra isteğe bağlı eval çalıştır
$env:GEMINI_API_KEY = "<Google AI Studio API key>"
python tr_duzelt.py --dosya girdi.txt --eval
```

`--eval` varsayılan olarak `gemini-4-hydrogen` kullanır. Bu seçenek verilmedikçe ağ isteği yapılmaz. Eval, hem girdinin tamamını hem düzeltilmiş çıktıyı Google Gemini API'ye gönderir. Google'ın ücretsiz kullanım koşullarında gönderilen içeriklerin ürünleri geliştirmek için kullanılabileceği ve insan incelemesine konu olabileceği belirtiliyor; **kişisel, gizli veya hassas metinleri göndermeyin**. Ücretsiz eval'i yalnızca hassas olmayan geliştirme/test girdilerinde kullanın ve kullanım koşullarını kontrol edin. Bkz. [Gemini API fiyatlandırma/veri kullanımı](https://ai.google.dev/gemini-api/docs/pricing) ve [Gemini API şartları](https://ai.google.dev/gemini-api/docs/terms).

Gemini yalnızca olası sorunları ve kural önerilerini raporlar; metni veya `GOVDE_DUZELT`/`TAM_DUZELT` sözlüklerini otomatik değiştirmez. Bir öneriyi ancak amaçlanan doğru metni elle doğruladıktan sonra kural tabanına ekleyin ve o davranış için regresyon testi yazın. Belirsiz/gürültülü girdilerde eval de hata yapabilir; sonucu kesin doğruluk ölçütü saymayın.

```python
from tr_duzelt import Duzeltici

d = Duzeltici("noktalama.crfsuite")      # yükleme ~15 sn (dil modeli + morfoloji)
print(d.duzelt("herkez geldimi bilmiyorum"))
```

`zemberek-python` saf Python olduğu için yavaştır (kelime başına kabaca 10–40 ms). Aynı mantık özgün Java Zemberek ile çok daha hızlı çalışır.

## Nasıl çalışır

| Katman | Teknik | Örnek |
|---|---|---|
| Yazım | Gürültülü kanal modeli: adaylar Zemberek'in morfolojik çözümleyicisi ve yazım denetleyicisinden, seçim 2-gram dil modeli üzerinde Viterbi ile | `zmaanlar → zamanlar`, `askari → askeri` |
| Konuşma dili | Zemberek'in gayriresmî morfolojisi + birkaç kural | `gidiyodum → gidiyordum`, `napcam → ne yapacağım` |
| Bitişik yazım | Her iki parçası da geçerli olan ayırmalar, dil modeliyle sınanır | `vebeni → ve beni`, `geldimi → geldi mi` |
| Sık yanlışlar | Küçük bir tablo (TDK'nin "sıkça yapılan yanlışlar" türünden) | `herkez → herkes`, `orjinal → orijinal` |
| Kurallar | Sayı ekleri, tarihler, ASCII metinde Türkçe karakterleri geri koyma | `2e → 2'ye`, `3 mayısta → 3 Mayıs'ta` |
| Noktalama + büyük harf | CRF dizi etiketleyici; öznitelikler: kelime/sonek + dil modelinin "bu kelimeden sonra nokta gelir mi" olasılığı ("gizli olay" dil modeli) | `... gittik film ...` → `... gittik. Film ...` |
| Özel isim | Zemberek çözümlemesi; çekim eki kesmeyle ayrılır, yapım eki ayrılmaz | `trabzona → Trabzon'a`, `ankaralı → Ankaralı` |

Noktalama modeli **kendinden denetimli** eğitilir: düzgün yazılmış metinden noktalama ve büyük harfler silinir, model silineni geri koymayı öğrenir. Elle etiketleme gerekmez.

## Ölçümler

**Yazım** — UD Turkish-IMST test bölümünden 603 cümle (5.681 kelime):

| Ölçüm | Sonuç |
|---|---|
| Tek harflik yapay hatayı geri getirme (500 cümle) | **%92,0** (460/500) |
| Aynı testte Zemberek'in hazır `TurkishSentenceNormalizer`'ı | %70,0 (350/500) |
| Temiz metinde dokunulan kelime oranı (yanlış alarm üst sınırı) | %0,32 (18/5.681) |

**Noktalama ve büyük harf** — F1, eğitimde hiç görülmemiş belgeler:

| Test kümesi | Nokta | Virgül | Soru işareti | Büyük harf |
|---|---|---|---|---|
| Gazete (TrMor2018, 21,6 bin kelime) | 0,806 | 0,332 | 0,580 | 0,784 |
| Roman (TrMor2018, 13,2 bin kelime) | 0,760 | 0,194 | 0,580 | 0,584 |
| Öğrenci yazıları, alan dışı (88 bin kelime) | 0,660 | 0,218 | 0,507 | – |
| Yalnız 2-gram dil modeli, CRF'siz (gazete / roman) | 0,782 / 0,705 | | | |

Kıyas için: aynı görev için ince ayarlanmış BERT tabanlı bir model, kendi test kümesinde nokta için 0,96, virgül için 0,63 F1 bildiriyor ([model kartı](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish)). Test kümeleri farklı olduğu için birebir karşılaştırma değildir, ama farkın yönü açık.

## Sınırlar

- **Yazımı doğru, anlamı yanlış kelimeler.** `bende/ben de`, `orda/orta`, `yaşam/yaşım` gibi durumlarda yazılan kelime de geçerli Türkçedir; doğrusunu seçmek anlam ister. 2-gram dil modeli bunu güvenle yapamıyor: yakalamak için eşiği düşürünce temiz metnin %0,77'sini bozuyor. Varsayılan ayar hiç bozmuyor (0/5.681) ama bu tür hataların yalnızca ~%20'sini yakalıyor.
- **Virgül.** F1 0,2–0,3. Yanlış virgül eksik virgülden daha rahatsız edici olduğu için model yalnızca çok emin olduğunda koyar.
- **Paragraf.** Denenen TextTiling benzeri yöntem, paragrafları belli 120 öğrenci yazısında rastgele bölmekten iyi çıkmadı (en uygun ayarında Pk 0,46–0,50; rastgele 0,49). Kodda duruyor ama varsayılan kapalı (`--paragraf`).
- **Devrik cümle, diyalog, liste.** Noktalama modeli düz anlatımda iyi, bu türlerde zayıf.
- **Bilinmeyen özel adlar.** Sözlükte olmayan yabancı ad ve markalar yanlışlıkla "düzeltilebilir". Girdide büyük harfle yazılmışlarsa korunurlar.

Noktalama ve anlam hataları için bir sonraki basamak üretken bir LLM değil, küçük bir encoder (BERT türü belirteç sınıflandırıcı) olur.

## Modeli yeniden eğitmek

```bash
python egit_noktalama.py --trmor2018 trmor2018.train --ud *.conllu \
    --cikti noktalama.crfsuite --test-kitap 14 17
```

Kendi alanınızdan düzgün yazılmış metinle eğitmek hem başarımı artırır hem lisans sorusunu sadeleştirir. Ayrıntılar `egit_noktalama.py` başlığında.

Windows'ta eğitirken `--cikti` yolunda Türkçe karakter olmasın: CRFsuite model dosyasını C kütüphanesiyle yazar ve bu tür yolları açamaz. (`tr_duzelt.py` modeli Python üzerinden okuduğu için bundan etkilenmez.)

## Veri, araçlar ve atıflar

`noktalama.crfsuite` şu derlemlerle eğitildi (≈858 bin kelime):

- [TrMor2018](https://github.com/ai-ku/TrMor2018) — Dayanık, Akyürek, Yüret (2018), MIT
- Universal Dependencies Türkçe ağaç yapıları: [BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN), [Kenet](https://github.com/UniversalDependencies/UD_Turkish-Kenet), [Penn](https://github.com/UniversalDependencies/UD_Turkish-Penn), [Tourism](https://github.com/UniversalDependencies/UD_Turkish-Tourism), [FrameNet](https://github.com/UniversalDependencies/UD_Turkish-FrameNet), [GB](https://github.com/UniversalDependencies/UD_Turkish-GB) (CC BY-SA 4.0), [PUD](https://github.com/UniversalDependencies/UD_Turkish-PUD) (CC BY-SA 3.0)

Model dosyası bu derlemlerden öğrenilmiş istatistikleri içerir; yeniden dağıtırken ya da ticari bir üründe kullanırken bu lisansların atıf ve aynı lisansla paylaşma koşullarını dikkate alın.

Yalnızca **ölçüm** için kullanılan, eğitime girmeyen ve bu depoda bulunmayan veriler:

- [UD Turkish-IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST) (CC BY-NC-SA 4.0) — yazım ölçümleri
- Bilkent Turkish Writings Dataset — Yilmaz, S. F. (2025). *Compilation of Bilkent Turkish Writings Dataset*. Zenodo. https://doi.org/10.5281/zenodo.15498155. Özgün kaynak: [Bilkent Üniversitesi Türkçe Birimi](https://stars.bilkent.edu.tr/turkce/). Öğrenci yazısı ve paragraf ölçümleri.

Kullanılan kütüphaneler: [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp) (Akın & Akın) ve Python uyarlaması [zemberek-python](https://github.com/Loodos/zemberek-python) — morfoloji, yazım denetleyici, dil modeli, deasciifier; [python-crfsuite](https://github.com/scrapinghub/python-crfsuite).

## Dayandığı çalışmalar

- Torunoğlu, D. & Eryiğit, G. (2014). A Cascaded Approach for Social Media Text Normalization of Turkish. *LASM @ EACL 2014*, 62–70. — kademeli normalizasyon mimarisi
- Kernighan, M., Church, K. & Gale, W. (1990). A Spelling Correction Program Based on a Noisy Channel Model. *COLING*.
- Mays, E., Damerau, F. & Mercer, R. (1991). Context Based Spelling Correction. *Information Processing & Management*. — gerçek-kelime hataları
- Stolcke, A. & Shriberg, E. (1996). Automatic Linguistic Segmentation of Conversational Speech. *ICSLP*. — gizli olay dil modeli
- Lafferty, J., McCallum, A. & Pereira, F. (2001). Conditional Random Fields. *ICML*.
- Hearst, M. (1997). TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages. *Computational Linguistics* 23(1).
