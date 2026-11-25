# CLAUDE.md — tr-duzelt devir notu

Bu dosya projeyi devralan Claude Code oturumu içindir. Kullanıcıya dönük anlatım `README.md`'de; burada yalnızca orada yazmayanlar var: sahibinin kuralları, tuzaklar, denenip bırakılanlar ve yarım kalan işler. Önce bunu, sonra README'yi oku. Bir işi bitirdiğinde bu dosyanın "Sıradaki işler" bölümünü güncelle.

Devir tarihi: 25 Kasım 2026. Devirden önceki son özellik commit'i: `4170084` (ardından yalnızca ölçüm yardımcıları ve bu dosya eklendi).

## Proje tek paragrafta

Bozuk yazılmış Türkçe metni düzeltir: yazım, Türkçe karakter, noktalama, büyük harf, kesme işareti. Çalışırken LLM kullanmaz. Hat: kelimelere ayır → her kelime için aday üret (kurallar + Zemberek) → öğrenilmiş sıralayıcı adayı seçer ya da kelimeyi korur → CRF noktalama koyar → büyük harf/kesme kuralları. Ayrıntı ve örnek döküm README'de ("Bir cümlenin yolculuğu").

## Sahibinin kuralları (bunlara uy)

1. **Çalışma anında LLM yok.** LLM yalnızca yardımcıdır: sonucu denetler (`--eval`, Gemini) ve kural önerir. Öneri elle doğrulanmadan koda girmez.
2. **ML ağırlıklı, üstüne büyüyen bir kural seti.** Doğruluk hızdan önemli; yerel makinede sürenin uzaması sorun değil.
3. **Emin değilsen dokunma.** Güvenle düzeltilemeyen kelime korunur; yanlış birleştirme/bölme ve anlamsız büyük harf en kötü hata sayılır (issue #2). Bir değişikliği değerlendirirken "bozulan doğru kelime" sütunu "düzeltilen"den önce gelir.
4. **Git:** yalnızca varsayılan dalda çalış, yan dal açma. Her özelliği ayrı commit yap ve hemen gönder. Commit mesajı Türkçe: ne değişti, neden, ölçümde ne oldu.
5. **README sade dille ve güncel.** "12 yaşındaki biri bile anlasın": ne nerede nasıl çalışıyor, bir cümlenin yolculuğu, hangi yöntem neden seçildi, hangisi neden seçilmedi; hepsi ölçülmüş sayılarla. Davranışı ya da sayıları değiştiren her commit README'yi de günceller.
6. **Windows, yönetici hakkı yok.** Kurulum önerirken yönetici onayı isteyen hiçbir adım kullanma (README'deki uv yolu).
7. **Gizlilik:** `--eval` metni Google'a gönderir; kişisel/gizli metinle çalıştırma. `GEMINI_API_KEY` değerini yazdırma, dosyaya ya da commit'e koyma.
8. Sahibi Türkçe yazar; kısa ve doğrudan cevap ister.

## Ortam ve komutlar

Python 3.11 ya da 3.12 (3.13 ve NumPy 2 ile `zemberek-python` çalışmaz; `requirements.txt` bunu sabitler). Windows'ta komutların başına `.venv\Scripts\` gelir.

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt   # kurulum
python tr_duzelt.py "herkez geldimi bilmiyorum"                # çalıştır
python tr_duzelt.py --yolculuk "..."                           # adım adım: hangi kelimeye neden dokunuldu
python -m unittest discover -s tests                           # testler (yalnızca tr_eval için var)
python olcum/olc.py                                            # ölçüm, bütün setler
python olcum/olc.py --sistem zemberek-normalizer --set gercek-test
python olcum/hata_turleri.py                                   # hata türüne göre: nerede kayıp var
python olcum/esik_tara.py                                      # sıralayıcı eşiğini tara
```

- Açılış 16–20 saniye sürer (Zemberek saf Python).
- `olcum/` ilk çalıştırmada verileri GitHub'dan `olcum/veri/` klasörüne indirir (gitignore'da). İlk ölçüm uzun sürer; Zemberek sonuçları `olcum/veri/zemberek-onbellek.pkl` dosyasına önbelleklenir, sonraki ölçümler dakikalar alır.
- Ek paketler yalnızca gerektiğinde: `lightgbm` (sıralayıcıyı eğitmek), `symspellpy` (SymSpell karşılaştırması), `torch transformers sentencepiece` (T5 karşılaştırması). `requirements.txt`'ye ekleme; düzelticinin kendisi bunları istemez.
- Windows'ta `lightgbm` içe aktarılamazsa büyük olasılıkla Visual C++ çalışma zamanı eksiktir; README'deki `msvc-runtime` çözümünü dene (denenmedi).

## Kod haritası

| Dosya | İçerik |
|---|---|
| `tr_duzelt.py` | Düzelticinin tamamı (aşağıda) |
| `siralayici.json` | Öğrenilmiş aday sıralayıcı: 50 öznitelik, 600 ağaç, eşik 0,5 |
| `noktalama.crfsuite` | CRF noktalama + büyük harf modeli |
| `egit_siralayici.py`, `egit_noktalama.py` | İki modelin eğitimi |
| `olcum/veri.py` | Test setlerini kurar (gerçek tweetler + yapay bozma); `Bozucu` eğitimde de kullanılır |
| `olcum/olcut.py` | Düzeltildi / bırakıldı / yanlış düzeltildi / bozuldu sayımı |
| `olcum/sistemler.py` | Karşılaştırılan yöntemler, aynı arayüzle (kelime listesi alır, kelime listesi verir) |
| `olcum/olc.py`, `hata_turleri.py`, `esik_tara.py` | Çalıştırıcılar |
| `tr_eval.py`, `tests/` | Sahibinin yazdığı Gemini denetimi ve testleri |

`tr_duzelt.py` içinde sırayla:

- `DilModeli`: Zemberek'in 2-gram dil modeli. `lp(a, b)` ikili olasılık, `var(a, b)` ikili derlemde görülmüş mü, `tekli(w)` sıklık, `ozel_isimler` derlemde özel isim gibi kullanılmış kökler.
- `crf_features`, `crf_coz`: noktalama modelinin öznitelikleri ve çözümü.
- Kural tabloları `GOVDE_DUZELT`, `TAM_DUZELT`, `KONUSMA_GECERLI`; maliyet sabitleri `C_*`; eşikler `GUVEN_*`, `KEYFI_DEGISIKLIK`, `C_BIRLESTIR_GECERLI`. Her sabitin yanında neden o değer olduğu yazıyor.
- `OZNITELIK_ADLARI`, `Siralayici`: JSON'daki ağaçları saf Python ile çalıştırır.
- `YazimDuzeltici`: `adaylar(w, genis)` aday üretir (`genis=True` gürültülü kaynakları da katar: Zemberek tabloları, geçerli kelimelerin aksanlı eşleri, de/da/ki ayırma). `birlesim_adaylari` bölünmüş kelimeler. `duzelt` sırasıyla: kafes üzerinde Viterbi → `_guven_denetimi` (elle ayarlı ilk geçiş) → `_siralayici_karari` (model varsa kararları onunkilerle değiştirir) → `_gercek_kelime`. `aday_satirlari` öznitelikleri çıkarır; eğitim de çalışma anı da aynı fonksiyonu kullanır.
- `Noktalayici`: CRF + `cozumle` (özel isim ve kesme işareti).
- `Duzeltici.duzelt`: bütün hat. `izle` listesi verilirse ara sonuçları kaydeder; `yolculuk_yaz` onu yazdırır.

## Değişiklik yaparken

Her gönderimden önce: `py_compile`, testler, bir `--yolculuk` denemesi, `python olcum/olc.py` ile gerileme kontrolü. Sayılar değiştiyse README tablolarını güncelle.

Tuzaklar:

- **Sıralayıcı öznitelikleri.** `OZNITELIK_ADLARI` ya da `aday_satirlari` değişirse `siralayici.json` yüklenirken `ValueError` verir. Yeniden eğit ve yeni modeli aynı commit'te gönder. Aday üretimi, maliyetler ya da tablolar değiştiğinde de model eski dağılımla eğitilmiş kalır: en azından ölç, fark varsa yeniden eğit.
- **Noktalama öznitelikleri.** `crf_features` eğitilmiş CRF ile birebir aynı kalmalı. Değiştirirsen `egit_noktalama.py` ile yeniden eğit.
- **Sözlük kuralları kesindir.** Tablolardan gelen düzeltmeyi sıralayıcı geri alamaz. Yalnızca her bağlamda doğru olan kuralı ekle (`direk → direkt` ekleme: "direk" geçerli kelime).
- **Geliştirme ve test ayrımı.** Tweetlerin ilk 1.000'i ayar içindir, son 1.000'i yalnızca sonucu bildirmek için. Eşik, kural ya da öznitelik seçerken teste bakma.
- **Veri lisansları.** #Turki$hTweets'in lisansı belirtilmemiş: yalnızca ölçüm ve eşik seçimi; eğitime sokma, depoya koyma. UD Turkish-IMST (CC BY-NC-SA) ve Bilkent yazıları (akademik kullanım) da yalnızca ölçüm içindir. Sıralayıcı UD Turkish-BOUN ile eğitilir.
- **Yapay testler iyimserdir.** Bozma yöntemi eğitimdekiyle aynı aileden. Asıl gösterge `gercek-test`.
- **Ölçüm yalnızca yazım aşamasını ölçer** (`YazimDuzeltici.duzelt`, küçük harfli kelimeler). Noktalama için bağımsız bir ölçüm betiği depoda yok; şimdilik `egit_noktalama.py --test-kitap 14 17` eğitimin sonunda ayrılmış belgelerde F1 yazdırıyor. README'deki "öğrenci yazıları" satırını üreten betik de depoda değil.
- **Windows:** CRFsuite model dosyasının yolunda Türkçe karakter olmasın (eğitirken). Çalışma anında sorun yok, model bellekten açılıyor.

Sıralayıcıyı yeniden eğitmek (depodaki model bu komutlarla üretildi; tablo dosyaları büyük, `olcum/veri/` altında tut):

```bash
pip install lightgbm
python egit_siralayici.py --cumle 2600 --tur 2 --tohum 2026 --yalniz-tablo --tablo olcum/veri/tablo_a.json
python egit_siralayici.py --cumle 2600 --atla 2600 --tur 2 --tohum 4711 --yalniz-tablo --tablo olcum/veri/tablo_b.json   # ayrı terminalde, aynı anda
python egit_siralayici.py --tablo olcum/veri/tablo_a.json olcum/veri/tablo_b.json   # eğitir, siralayici.json yazar
python olcum/esik_tara.py && python olcum/olc.py
```

Tablo üretimi iki süreçle 15–25 dakika, tek süreçle (`python egit_siralayici.py`) yaklaşık bir saat sürer. Sonuç bire bir aynı çıkmayabilir, ölçüm gürültüsü kadar oynar.

## Son ölçüm

Gerçek tweetler, test bölümü (1.000 tweet, 2.226 hata):

| Yöntem | Düzeltilen | Yanlış düzeltilen | Bozulan doğru kelime | F0.5 |
|---|---|---|---|---|
| Zemberek yazım denetleyicisi, bağlamsız | %33,5 | %28,6 | %1,47 | 46,0 |
| SymSpell | %42,0 | %45,1 | %11,82 | 36,6 |
| T5 (`cguzel/trspell-t5-efficient-small`) | %19,2 | %8,2 | %1,27 | 43,1 |
| Zemberek hazır normalleştirici | %75,0 | %11,6 | %2,44 | 79,4 |
| tr-duzelt ilk sürüm | %62,8 | %10,7 | %1,39 | 76,8 |
| tr-duzelt, sıralayıcısız (`tr-duzelt-sezgisel`) | %67,3 | %5,9 | %0,45 | 84,7 |
| **tr-duzelt, şimdiki** | **%73,7** | **%4,1** | **%0,42** | **88,6** |

Aynı model geliştirme bölümünde: %73,9 / %3,9 / %0,39 / 89,0. Yapay setler: tek hata %86,4 düzeltilen, %4,1 yanlış; yoğun gürültü %86,9 / %3,2; bölünmüş kelime %88,2 / %3,3; araya giren İngilizcede bozma %0,16; temiz metinde %0,18.

Hata türüne göre düzeltilen oran (geliştirme): Türkçe karakter eksikliği %93, konuşma dili %73, bilerek uzatma/kısaltma %62, ayrı yazılması gereken %56, yazım hatası %50.

## Denenip bırakılanlar (yeniden deneme, nedenini bil)

- **Zemberek'in 470 bin satırlık tablolarını elle kuralla aday yapmak:** 32 doğru düzeltmeye karşı 74 yanlış. "Yalnızca başka aday yoksa" kullanınca da kazanç çıkmadı. Sıralayıcının girdisi olarak işe yarıyor (`genis=True`).
- **Deasciifier'ı ön adım yapmak:** Türkçe karaktersiz 364 tweette bozmayı %0,40'tan %2,13'e çıkardı (`go → gö`). Şimdi yalnızca öznitelik.
- **3 harf ve altındaki bilinmeyen kelimeye harf tahmini:** İngilizce işlev kelimelerini Türkçeye çeviriyor (the, you, and).
- **Çift harfi her yerde uzatma saymak:** `mood → mod`.
- **Birleştirmede tek harfli parça ya da ve/de/mi gibi işlev kelimesi:** `i de → ide`, `bro ve → bröve`.
- **Paragraf bölme:** rastgele bölmekten iyi çıkmadı; kodda kapalı duruyor (`--paragraf`).
- **T5'i ayrı yöntem ya da aday kaynağı olarak kullanmak:** yukarıda, "Sıradaki işler" 2–3.
- **Pekiştirmeli öğrenme:** elimizde "bozuk → doğru" çiftleri olduğu için denetimli öğrenme daha verimli; gerek görülmedi.

## Sıradaki işler

1. **Depo temizliği (sahibi istedi, önceki oturumun GitHub erişimi izin vermedi).** `windows-kurulum` ve `issue-2-temkinli-duzeltme` dalları tamamen `main`'in içinde; sil. Sonra `main` → `master` yap:
   ```bash
   git push origin --delete windows-kurulum issue-2-temkinli-duzeltme
   gh api -X POST repos/oguzhanozfe/tr-duzelt/branches/main/rename -f new_name=master
   git branch -m main master && git fetch origin && git branch -u origin/master master
   ```
   `gh` yoksa ad değişikliği GitHub'da Settings → Branches'tan yapılır. README'deki indirme bağlantısı dal adından bağımsız; başka değişiklik gerekmez.
2. ~~T5 ölçümü~~ — yapıldı (25 Kasım 2026): test %19,2 / %8,2 / %1,27. `T5Yazim` ekran kartını kullanıyor.
3. ~~T5'i aday kaynağı yapmak~~ — üst sınır ölçüldü, yapılmadı: geliştirmede tr-duzelt'in kaçırdığı 324 kelimeden (eşit uzunluklu 723 cümle) T5 yalnızca 48'ini doğru veriyor; yanlış düzeltmeyi azaltmaz, çalışma anına torch + 600 MB getirir.
4. **Noktalama için BERT.** En büyük kazanç beklenen yer; CRF'in virgül F1'i 0,19–0,33. Önce bağımsız bir noktalama ölçüm betiği yaz (`olcum/` altına), sonra hazır bir modeli (`uygarkurt/bert-restore-punctuation-turkish`) ya da BERTurk üzerine eğitilmiş bir etiketleyiciyi aynı veride CRF ile karşılaştır. Çalışma anına `torch` bağımlılığı ekleyeceği için isteğe bağlı kalmalı; kararı sahibine sor.
5. **Bağlam için daha güçlü dil modeli.** Şimdiki model yalnızca yandaki kelimeye bakıyor. BERT'in maskeli dil modeli puanını sıralayıcıya öznitelik olarak eklemek `yaşam/yaşım`, `de/da` gibi durumlara yarayabilir.
6. **Kural büyütme döngüsü.** `--eval` çıktısındaki "olası kural" önerileri → elle doğrula → tabloya ekle → `olcum/olc.py`. İstenirse tablolar ayrı bir dosyaya taşınıp, önerilen kuralı otomatik sınayan bir betik yazılabilir (yanlış biçim Zemberek'te geçersiz mi, doğrusu geçerli mi, temiz metinde bozma artıyor mu).
7. **Sahibine sorulacaklar:** depoda `LICENSE` yok; depo açıklaması boş. Issue #2 açık, özet yorumu yazıldı; sahibi kendi makinesinde deneyip kapatacak.

## Bilinen hatalar

- Issue #2 cümlesinde kalanlar: `sasırdım` (Zemberek geçerli sayıyor), `dokmusa` ve `mekanası` (aday yok), `dağ dan` (iki parça da geçerli kelime, dil modeli kazancı eşiğin altında), devrik cümlede nokta yeri ("Aslan kaçtı. Ormana…").
- `yaaa` gibi yalnızca özel isim olarak "geçerli" sayılan uzatmalar çoğu zaman kısaltılmıyor.
- Zemberek'in önerileri bir harf mesafeyle sınırlı; iki harfi yanlış kelime için aday çıkmıyor.
- Zemberek'in özel isim sözlüğü çok geniş (`we`, `go`, `na`, `boyle` bile var). `gecerli()` bu yüzden fazla hoşgörülü; sıradan kelime mi diye bakarken `duzgun()` kullan.
- Bitişik yazılmış de/da/ki hatalarının yaklaşık yarısı düzeliyor; temiz metinde nadiren yanlış ayırma da yapıyor (`etkilemede → etkileme de`).
- "Bozulan" sayılanların bir kısmı aslında doğru düzeltme (`lüften → lütfen`); tweet verisini düzeltenler bunlara dokunmamış. Gerçek bozma oranı ölçülenden biraz düşük.
