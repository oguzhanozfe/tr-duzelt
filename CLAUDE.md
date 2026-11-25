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
| `siralayici.json` | Öğrenilmiş aday sıralayıcı: 50 öznitelik, eşik 0,7 (25 Kasım'da dev'de seçildi; model aynı) |
| `siralayici_bert.json` | `--bert` sıralayıcısı: 50 + 5 BERT özniteliği, eşik 0,5 |
| `noktalama_bert/` | `--bert` noktalama modeli (BERTurk, 211 MB, depoda DEĞİL, .gitignore'da); `egit_noktalama_bert.py` üretir |
| `ornekler/` | Kendi yazdığımız 20 metin + iki kipin çıktısı (README'deki örnekler buradan) |
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
- **Ölçüm yalnızca yazım aşamasını ölçer** (`YazimDuzeltici.duzelt`, küçük harfli kelimeler). Noktalama `olcum/noktalama_olc.py` ile ayrıca ölçülür (UD Turkish-IMST dev+test; hiçbir model bununla eğitilmedi). `egit_noktalama.py --test-kitap 14 17` de eğitimin sonunda ayrılmış TrMor belgelerinde F1 yazdırıyor. README'deki "öğrenci yazıları" satırını üreten betik de depoda değil.
- **Windows:** CRFsuite model dosyasının yolunda Türkçe karakter olmasın (eğitirken). Çalışma anında sorun yok, model bellekten açılıyor.
- **Windows'ta torch ve uzun yol:** venv çok uzun bir klasördeyse `torch` DLL'leri yüklenmez (`WinError 206`). 25 Kasım oturumunda torch'lu ortam kısa yolda kuruldu (`C:\Users\<kullanıcı>\.venvs\trd`).
- **GPU'yu gizlemek:** Windows'ta `CUDA_VISIBLE_DEVICES=` (boş) işe yaramıyor, `CUDA_VISIBLE_DEVICES=-1` kullan. İşlemci hızını ölçerken buna dikkat et.
- **BERT tablo üretimi:** 4 süreç + `OMP_NUM_THREADS=4` sorunsuz (~5 dk). 8 süreç 24 GB ekran belleğini taşırdı ve süreçler saatlerce süründü. Tablo üretimiyle noktalama eğitimini aynı anda çalıştırma.
- **BERT öznitelikleri:** `BertPuan.PENCERE` (adayın iki yanındaki alt parça sayısı) eğitimde ve çalışma anında aynı olmalı (şimdi 40). Değişirse tabloyu yeniden üret. BERT'li tablonun ilk 50 sütunu BERT'siz tablodur: `egit_siralayici.py --tablo ...` (`--bert` olmadan) aynı veriden taban modeli eğitir.
- **Kural son geçişi** (`_kural_son_gecis`) yalnızca zaten değiştirilmiş kelimelere uygulanır; sıralayıcıdan sonra çalıştığı için tabloları bayatlatmaz. Temiz metni korumak için bu koşulu gevşetme (yazarın kendi "hiç bir"i korunur).

Sıralayıcıyı yeniden eğitmek (depodaki model bu komutlarla üretildi; tablo dosyaları büyük, `olcum/veri/` altında tut):

```bash
pip install lightgbm
python egit_siralayici.py --cumle 2600 --tur 2 --tohum 2026 --yalniz-tablo --tablo olcum/veri/tablo_a.json
python egit_siralayici.py --cumle 2600 --atla 2600 --tur 2 --tohum 4711 --yalniz-tablo --tablo olcum/veri/tablo_b.json   # ayrı terminalde, aynı anda
python egit_siralayici.py --tablo olcum/veri/tablo_a.json olcum/veri/tablo_b.json   # eğitir, siralayici.json yazar
python olcum/esik_tara.py && python olcum/olc.py
```

Tablo üretimi iki süreçle 15–25 dakika, tek süreçle (`python egit_siralayici.py`) yaklaşık bir saat sürer. Sonuç bire bir aynı çıkmayabilir, ölçüm gürültüsü kadar oynar.

## Son ölçüm (25 Kasım 2026, ikinci oturum)

Gerçek tweetler, test bölümü (1.000 tweet, 2.226 hata):

| Yöntem | Düzeltilen | Yanlış düzeltilen | Bozulan doğru kelime | F0.5 |
|---|---|---|---|---|
| Zemberek yazım denetleyicisi, bağlamsız | %33,5 | %28,6 | %1,47 | 46,0 |
| SymSpell | %42,0 | %45,1 | %11,82 | 36,6 |
| T5 (`cguzel/trspell-t5-efficient-small`) | %19,2 | %8,2 | %1,27 | 43,1 |
| Zemberek hazır normalleştirici | %75,0 | %11,6 | %2,44 | 79,4 |
| tr-duzelt ilk sürüm | %62,8 | %10,7 | %1,39 | 76,8 |
| tr-duzelt, sıralayıcısız (`tr-duzelt-sezgisel`) | %67,3 | %5,9 | %0,45 | 84,7 |
| tr-duzelt, devir anı (`4170084`) | %73,7 | %4,1 | %0,42 | 88,6 |
| **tr-duzelt, varsayılan (eşik 0,7)** | **%72,7** | **%2,7** | **%0,27** | **89,9** |
| **tr-duzelt `--bert`** | **%75,7** | **%2,7** | **%0,35** | **90,7** |
| `--temkinli` (0,9) | %66,0 | %1,9 | %0,12 | 88,5 |
| `--bert --temkinli` | %68,9 | %1,8 | %0,13 | 89,6 |

Geliştirme: varsayılan %72,3 / %2,4 / %0,36; `--bert` %74,6 / %2,9 / %0,26. Yapay setler (varsayılan / `--bert`): tek hata %83,8 / %89,0 düzeltilen; üç hata %85,0 / %90,1; yoğun %84,9 / %88,3; temiz metinde bozma %0,12 / %0,14; İngilizcede %0,11 / %0,16.

Noktalama, IMST test F1 (`olcum/noktalama_olc.py`): CRF nokta 0,675 / virgül 0,079 / soru 0,341; `noktalama_bert` (virgül ≥0,7, soru ≥0,6) 0,782 / 0,433 / 0,463, virgül isabeti 0,747.

Hız (yalnızca işlemci, 32 çekirdek): `--toplu` varsayılan 8 süreç 24 satır/sn, 3 süreç 11, `--bert` 8 süreç 17 (2.000 tweet, 8,5 kelime/satır). Tek iş parçacığında kelime seçimi: BERT'siz ~440, BERT'li kapılı int8 ~65 kelime/sn; noktalama BERT ~300.

Bu oturumun commit'leri sırayla: `948a8f7` T5, `95eecd0` noktalama ölçümü, `884582e` + `c0810b1` kural son geçişi, `111223a` aday üretimi + eşik 0,7, `d584c31` BERT kipi, `fb53b0f` toplu mod.

## Denenip bırakılanlar (yeniden deneme, nedenini bil)

- **Zemberek'in 470 bin satırlık tablolarını elle kuralla aday yapmak:** 32 doğru düzeltmeye karşı 74 yanlış. "Yalnızca başka aday yoksa" kullanınca da kazanç çıkmadı. Sıralayıcının girdisi olarak işe yarıyor (`genis=True`).
- **Deasciifier'ı ön adım yapmak:** Türkçe karaktersiz 364 tweette bozmayı %0,40'tan %2,13'e çıkardı (`go → gö`). Şimdi yalnızca öznitelik.
- **3 harf ve altındaki bilinmeyen kelimeye harf tahmini:** İngilizce işlev kelimelerini Türkçeye çeviriyor (the, you, and).
- **Çift harfi her yerde uzatma saymak:** `mood → mod`.
- **Birleştirmede tek harfli parça ya da ve/de/mi gibi işlev kelimesi:** `i de → ide`, `bro ve → bröve`.
- **Paragraf bölme:** rastgele bölmekten iyi çıkmadı; kodda kapalı duruyor (`--paragraf`).
- **T5'i ayrı yöntem ya da aday kaynağı olarak kullanmak:** test %19,2 / %8,2 / %1,27; aday kaynağı üst sınırı: tr-duzelt'in kaçırdığı 324 kelimeden 48'i.
- **Hazır noktalama BERT'i** (`uygarkurt/bert-restore-punctuation-turkish`): cümle sonlarına virgül koyuyor; IMST nokta F1 0,276.
- **BERTurk uncased** (`dbmdz/bert-base-turkish-128k-uncased`): Türkçe karakterleri siliyor (çok -> cok), PLL ayırt edemiyor. Cased kullan.
- **BERT'i bütün kelimelere sormak:** işlemcide 13 kelime/sn; kapı (taban olasılığı 0,05-0,98 arası) ile 39, int8 ile 65; doğruluk aynı.
- **Noktalama BERT'inde int8:** 2 kat hızlı ama virgül F1 0,433 -> 0,400; zaten hızlı olduğu için fp32.
- **Taban sıralayıcıyı yeni tablolarla yeniden eğitmek:** geliştirmede gürültü düzeyinde (0,5'te %74,6 / %3,4 / %0,37, eski model %74,9 / %3,1 / %0,40); eski model tutuldu.
- **Eşiği tek başına yükseltmek:** 0,95'te yanlış %2,1 ama düzeltilen %60; kural geçişi + eşik 0,7 daha iyi denge.
- **Pekiştirmeli öğrenme:** elimizde "bozuk → doğru" çiftleri olduğu için denetimli öğrenme daha verimli; gerek görülmedi.

## Sıradaki işler

1. **Depo temizliği (sahibi istedi, önceki oturumun GitHub erişimi izin vermedi).** `windows-kurulum` ve `issue-2-temkinli-duzeltme` dalları tamamen `main`'in içinde; sil. Sonra `main` → `master` yap:
   ```bash
   git push origin --delete windows-kurulum issue-2-temkinli-duzeltme
   gh api -X POST repos/oguzhanozfe/tr-duzelt/branches/main/rename -f new_name=master
   git branch -m main master && git fetch origin && git branch -u origin/master master
   ```
   `gh` yoksa ad değişikliği GitHub'da Settings → Branches'tan yapılır. README'deki indirme bağlantısı dal adından bağımsız; başka değişiklik gerekmez.
2. ~~T5 ölçümü~~, 3. ~~T5 aday kaynağı~~ — yapıldı, değmedi (yukarıda).
4. ~~Noktalama için BERT~~ — yapıldı (`--bert`, `noktalama_bert/`). **Açık soru:** 211 MB'lık model depoda değil; GitHub sürümü (release) olarak mı yayımlansın, Hugging Face'e mi konsun, yoksa herkes kendisi mi eğitsin? Sahibine sor.
5. ~~BERT maskeli dil modeli puanı öznitelik~~ — yapıldı (`siralayici_bert.json`, kapılı).
6. **Kural büyütme döngüsü.** `--eval` çıktısındaki "olası kural" önerileri → elle doğrula → tabloya ekle → `olcum/olc.py`. İstenirse tablolar ayrı bir dosyaya taşınıp, önerilen kuralı otomatik sınayan bir betik yazılabilir (yanlış biçim Zemberek'te geçersiz mi, doğrusu geçerli mi, temiz metinde bozma artıyor mu).
7. **Gelecek zamanın konuşma biçimleri.** `başlicak` (başlayacak), `yagicak` (yağacak), `alcam` (alacağım) çoğu zaman düzelmiyor: ünlü düşmüş/daralmış gelecek zaman için aday üretilmiyor. `konusma_dili`'ne "c/ç'den önceki ı/i'yi a/e(y)a ile değiştir, ünlü yoksa ekle" biçimleri eklenebilir; aday tarafı olduğu için sonra iki sıralayıcıyı da yeniden eğit ve ölç.
8. **Kalan yanlış düzeltmelerin bir kısmı ölçüt kaynaklı:** altın veri "bi cok" → "birçok" gibi birleştirmeleri tek düzenleme sayıyor; "bi -> bir" kısmi düzeltmesi yanlış sayılıyor. `../logs/yanlis_dev*.txt` türü dökümü `olcum/` altına betik olarak almak işe yarar.
9. **Sahibine sorulacaklar:** depoda `LICENSE` yok; depo açıklaması boş. Issue #2 açık, özet yorumu yazıldı; sahibi kendi makinesinde deneyip kapatacak.

## Bilinen hatalar

- Issue #2 cümlesinde kalanlar: `sasırdım` (Zemberek geçerli sayıyor), `dokmusa` ve `mekanası` (aday yok), `dağ dan` (iki parça da geçerli kelime, dil modeli kazancı eşiğin altında), devrik cümlede nokta yeri ("Aslan kaçtı. Ormana…").
- `yaaa` gibi yalnızca özel isim olarak "geçerli" sayılan uzatmalar çoğu zaman kısaltılmıyor.
- Zemberek'in önerileri bir harf mesafeyle sınırlı; iki harfi yanlış kelime için aday çıkmıyor.
- Zemberek'in özel isim sözlüğü çok geniş (`we`, `go`, `na`, `boyle` bile var). `gecerli()` bu yüzden fazla hoşgörülü; sıradan kelime mi diye bakarken `duzgun()` kullan.
- Bitişik yazılmış de/da/ki hatalarının yaklaşık yarısı düzeliyor; temiz metinde nadiren yanlış ayırma da yapıyor (`etkilemede → etkileme de`).
- "Bozulan" sayılanların bir kısmı aslında doğru düzeltme (`lüften → lütfen`); tweet verisini düzeltenler bunlara dokunmamış. Gerçek bozma oranı ölçülenden biraz düşük.
