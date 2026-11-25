# tr-duzelt

Bozuk yazılmış Türkçe metni düzeltir: **yazım, Türkçe karakterler, noktalama, büyük harf, kesme işareti**.

```
Girdi : bugun 3 mayista ankarada bulustuk tatlımmm cok guzeldi tmm mı
Çıktı : Bugün 3 Mayıs'ta Ankara'da buluştuk. Tatlım çok güzeldi. Tamam mı?
```

Bunu yaparken **yapay zekâ sohbet modeli (LLM) kullanmaz**. Her şey kendi bilgisayarınızda çalışır, internet ve ekran kartı gerekmez, kelime başına para (token) harcanmaz. İçinde iki küçük "öğrenmiş" model ve bir avuç kural var; hepsi aşağıda tek tek anlatılıyor.

En önemli ilkesi: **emin değilse dokunmaz.** Yanlış bir kelimeyi olduğu gibi bırakmak, doğru bir kelimeyi bozmaktan iyidir.

Bu bir prototiptir. Neyi ne kadar yapabildiği ve **yapamadığı** ölçümleriyle yazıyor.

## İçindekiler

1. [Örnekler](#örnekler)
2. [Bir cümlenin yolculuğu](#bir-cümlenin-yolculuğu) — ne, nerede, nasıl çalışıyor
3. [Neden bu yöntemler, neden ötekiler değil?](#neden-bu-yöntemler-neden-ötekiler-değil)
4. [Ne kadar iyi?](#ne-kadar-iyi) — ölçümler
5. [Yapamadıkları](#yapamadıkları)
6. [Kural eklemek ve LLM'in rolü](#kural-eklemek-ve-llmin-rolü)
7. [Kurulum](#kurulum) · [Kullanım](#kullanım) · [Dosyalar](#dosyalar-ne-nerede)
8. [Modelleri yeniden eğitmek](#modelleri-yeniden-eğitmek) · [Veri ve atıflar](#veri-araçlar-ve-atıflar)

## Örnekler

Çıktılar bu depodaki kod ve modellerle üretilmiştir.

| Girdi | Çıktı |
|---|---|
| bugün okula gitmedim çünki hastaydım annem beni doktora götürdü doktor bişey yok dedi ama ben hala kötü hissediyorum yarın sınavım var napcam bilmiyorum ankarada hava çok soğuk | Bugün okula gitmedim çünkü hastaydım. Annem beni doktora götürdü. Doktor bir şey yok dedi. Ama ben hala kötü hissediyorum. Yarın sınavım var ne yapacağım bilmiyorum. Ankara'da hava çok soğuk. |
| kardeşim 3 mayısta doğdu o zamanlar izmirde yaşıyorduk babam bir fabrikada çalışıyordu | Kardeşim 3 Mayıs'ta doğdu. O zamanlar İzmir'de yaşıyorduk. Babam bir fabrikada çalışıyordu. |
| herkez 1994de doğmuş gibi davranıyo yanlız ben 2000li yılların çocuğuyum | Herkes 1994'te doğmuş gibi davranıyor. Yalnız ben 2000'li yılların çocuğuyum. |
| dun aksam arkadaslarimla sinemaya gittik film cok guzeldi sonra yemek yedik eve gec geldim | Dün akşam arkadaşlarımla sinemaya gittik. Film çok güzeldi. Sonra yemek yedik. Eve geç geldim. |
| dün arkadaşım ahmetle buluştuk o bana yeni telefonunu gösterdi çok beğendim bende almak istiyorum ama param yok | Dün arkadaşım Ahmet'le buluştuk. O bana yeni telefonunu gösterdi. Çok beğendim. Ben de almak istiyorum ama param yok. |
| benım arkadasım dıye geldıııı ama cooook yoruldummm yaaa, boyle olmaz kii | Benim arkadaşım diye geldi ama çok yoruldum. Yaaa, böyle olmaz ki. |
| bu aksam mac var here we go again diyoruz ama cringe olmasin bro | Bu akşam maç var. Here we go again diyoruz ama cringe olmasın bro. |

Son iki satır düzelticinin huyunu gösteriyor: `yaaa` kısaltılmadı çünkü emin olamadı; İngilizce kısma hiç dokunmadı.

## Bir cümlenin yolculuğu

Bir metin düzelticiye girince sırayla dört duraktan geçer. Aşağıdaki döküm gerçek çıktıdır; kendi cümleniz için aynısını şu komut verir:

```bash
python tr_duzelt.py --yolculuk "bugun 3 mayista ankarada bulustuk tatlımmm cok guzeldi tmm mı"
```

```
GİRDİ
  bugun 3 mayista ankarada bulustuk tatlımmm cok guzeldi tmm mı

1. DURAK: kelimelere ayır (10 kelime)
  bugun | 3 | mayista | ankarada | bulustuk | tatlımmm | cok | guzeldi | tmm | mı

2. DURAK: her kelime için adaylar üret, en uygununu seç
  (yalnızca birden çok seçeneği olan ya da tanınmayan kelimeler gösteriliyor)
  bugun            -> bugün              [aksan]  adaylar: bugün (aksan)
  mayista          -> mayıs'ta           [yazım]  adaylar: mayısta (aksan), mayıs'ta (yazım), mayıs da (ayırma)
  bulustuk         -> buluştuk           [aksan]  adaylar: buluştuk (aksan)
  tatlımmm         -> tatlım             [konuşma dili]  adaylar: tatlım (konuşma dili)
  cok              -> çok                [aksan]  adaylar: çok (aksan), çök (tablo)
  guzeldi          -> güzeldi            [aksan]  adaylar: güzeldi (aksan)
  tmm              -> tamam              [sözlük]  adaylar: tamam (sözlük), taam (tablo), tam (tablo)

3. DURAK: noktalama ([.] [,] [?] = modelin eklediği işaret)
  bugün 3 mayıs'ta ankarada buluştuk[.] tatlım çok güzeldi[.] tamam mı[?]

4. DURAK: büyük harf ve kesme işareti
  bugün -> Bugün, mayıs'ta -> Mayıs'ta, ankarada -> Ankara'da, tatlım -> Tatlım, tamam -> Tamam

ÇIKTI
  Bugün 3 Mayıs'ta Ankara'da buluştuk. Tatlım çok güzeldi. Tamam mı?
```

Şimdi durakları tek tek açalım.

### 1. durak: kelimelere ayır

Metin boşluklardan ve noktalama işaretlerinden bölünür. Bu kadar basit. Sonraki duraklar hep kelime kelime çalışır.

### 2. durak: her kelime için aday üret, sonra seç

Bu durak iki işe ayrılır: önce "bu kelime aslında ne olabilir?" diye **adaylar** toplanır, sonra adaylardan biri **seçilir** (ya da kelime olduğu gibi bırakılır).

**Adaylar nereden gelir?** Yedi ayrı kaynaktan. Dökümde her adayın yanında parantez içinde kaynağı yazar.

| Kaynak | Ne yapar | Örnek |
|---|---|---|
| **sözlük** | Elle yazılmış "yanlış → doğru" listesi. En güvenilir kaynak. | `herkez → herkes`, `tmm → tamam` |
| **aksan** | Eksik Türkçe karakterleri (ç ğ ı ö ş ü) dener. | `bugun → bugün`, `benım → benim` |
| **konuşma dili** | Konuşur gibi yazılanı yazı diline çevirir, uzatılmış harfleri kısaltır. | `gelcektim → gelecektim`, `tatlımmm → tatlım` |
| **yazım** | Bir harf eksik, fazla ya da yanlış yazılmış kelimeler için en yakın geçerli kelimeler. | `bilemiyrum → bilemiyorum` |
| **tablo** | Zemberek'in milyonlarca cümleden kendiliğinden çıkardığı 470 bin satırlık liste. Çok şey bilir ama sık yanılır. | `msj → mesaj` (ama `tmm → tam` da der) |
| **ayırma** | Bitişik yazılmış iki kelimeyi ayırır. | `vebeni → ve beni`, `bende → ben de` |
| **birleştirme** | Yanlışlıkla bölünmüş kelimeyi birleştirir. | `orma na → ormana` |

Buradaki gizli kahraman **Zemberek**: Türkçe bir kelimeyi köküne ve eklerine ayırabilen açık kaynak bir kütüphane. Türkçede bir kökten binlerce kelime türer (`git`, `gittim`, `gidiyordum`, `gidemeyeceklermiş`…), bu yüzden "doğru kelimelerin listesi" diye bir şey tutulamaz. Zemberek listeye bakmaz, kelimeyi çözer: `gidiyordum = git + iyor + du + m`. Çözebiliyorsa kelime geçerlidir.

**Seçimi kim yapar?** Küçük bir öğrenmiş model: **sıralayıcı** (`siralayici.json`). Her adaya bakar ve tek bir soruya cevap verir: *"Yazan kişi bunu mu demek istedi?"* Cevabı 0 ile 1 arasında bir olasılıktır. Karar verirken 50 ipucuna bakar, örneğin:

- Aday, yazılan kelimeye ne kadar benziyor? Fark tek harf mi, yalnızca Türkçe karakter mi?
- Aday Türkçede ne kadar sık kullanılıyor?
- Aday, yanındaki kelimelerle birlikte daha önce görülmüş mü? ("dün akşam" çok görülür, "dün aksam" az.)
- Cümlenin geri kalanı nasıl? Hiç Türkçe karakter yoksa büyük olasılıkla hepsi eksik yazılmıştır.
- Yanındaki kelimeler de tanınmıyor mu? O zaman bu bir yabancı cümle olabilir, dokunma.

Bir aday ancak olasılığı hem 0,5'i hem de "olduğu gibi bırak" seçeneğini geçerse uygulanır. Geçemezse kelime **korunur**. İkinci örnekte bunu göreceksiniz.

Sıralayıcı bunları nereden öğrendi? 5.200 düzgün cümle aldık ve onları insanların yaptığı gibi **bile bile bozduk**: harf sildik, Türkçe karakterleri attık, harfleri uzattık, kelimeleri birleştirdik, araya İngilizce koyduk. Doğrusunu bildiğimiz için her adayın "doğru" ya da "yanlış" olduğunu da biliyorduk. Model 125 bin böyle örneğe bakarak hangi ipuçlarının neyi gösterdiğini çıkardı. (Yöntemin adı: gradyan artırmalı karar ağaçları. Ardı ardına sorulan yüzlerce küçük "evet/hayır" sorusu gibi düşünün.)

Bir istisna: **sözlük** kaynağından gelen düzeltme kesindir, sıralayıcı onu geri alamaz. Böylece eklediğiniz her kural her zaman çalışır.

### 3. durak: noktalama

Kelimeler düzeldikten sonra ikinci öğrenmiş model devreye girer: **noktalama modeli** (`noktalama.crfsuite`). Her kelimeden sonra şuna karar verir: nokta mı gelir, virgül mü, soru işareti mi, hiçbiri mi?

Bunu da örnekten öğrendi: 858 bin kelimelik düzgün yazılmış metinden noktalama işaretlerini sildik, model silinenleri geri koymayı öğrendi. Kimse elle işaretlemedi. Baktığı ipuçları: kelimenin kendisi ve sonu (`-dı`, `-yor`, `-mış` ile biten kelimeden sonra çoğu zaman nokta gelir), komşu kelimeler, cümlenin neresinde olduğu. (Yöntemin adı: CRF. Kelimelere tek tek değil, birbirini etkileyen bir sıra olarak bakan bir model.)

Yazarın kendi koyduğu noktalama işaretlerine dokunulmaz; model yalnızca eksikleri tamamlar.

### 4. durak: büyük harf ve kesme işareti

- **Cümle başı** büyük yazılır.
- **Özel isimler** büyük yazılır ve ekleri kesme işaretiyle ayrılır: `ankarada → Ankara'da`. Ama bunun için üç kanıt birlikte aranır: Zemberek kelimeyi özel isim olarak tanımalı, kelime büyük bir metin yığınında gerçekten özel isim gibi kullanılmış olmalı, ve kelimenin başka anlamı da varsa (`deniz`, `gül`) noktalama modeli bağlamdan "bu özel isim" demeli. Tek kanıt yetmez; yoksa `here we go again` → `Here We Go Again` olur.
- **Tarihler ve sayılar** kuralla düzeltilir: `3 mayısta → 3 Mayıs'ta`, `1994de → 1994'te`, `2e → 2'ye`.
- Yazarın **kendi yazdığı büyük harf** her zaman korunur.

### İkinci yolculuk: çok bozuk bir cümle

Bu örnek, düzelticinin neye **dokunmadığını** gösteriyor.

```
GİRDİ
  aslan kacti orma na son ra bulduk vwovv oldum here we go again dedik

1. DURAK: kelimelere ayır (14 kelime)
  aslan | kacti | orma | na | son | ra | bulduk | vwovv | oldum | here | we | go | again | dedik

2. DURAK: her kelime için adaylar üret, en uygununu seç
  (yalnızca birden çok seçeneği olan ya da tanınmayan kelimeler gösteriliyor)
  kacti            -> kaçtı              [tablo]  adaylar: kaçtı (tablo), kati (yazım), kasti (yazım)
  orma na          -> ormana             [birleştirme: iki parça tek kelime]
  son ra           -> sonra              [birleştirme: iki parça tek kelime]
  oldum            =  korundu            adaylar: öldüm (aksan)

3. DURAK: noktalama ([.] [,] [?] = modelin eklediği işaret)
  aslan kaçtı[.] ormana sonra bulduk[.] vwovv oldum[.] here we go again dedik[.]

4. DURAK: büyük harf ve kesme işareti
  aslan -> Aslan, ormana -> Ormana, vwovv -> Vwovv, here -> Here

ÇIKTI
  Aslan kaçtı. Ormana sonra bulduk. Vwovv oldum. Here we go again dedik.
```

- `orma na` ve `son ra` birleştirildi. İki parça da tek başına kelime değilse ve birleşince geçerli bir kelime çıkıyorsa bu güvenli bir düzeltmedir.
- `vwovv` için hiç aday yok; olduğu gibi kaldı.
- `oldum` için `öldüm` adayı vardı ama sıralayıcı ikna olmadı; kelime korundu. Doğrusu da bu.
- `here we go again` yabancı bir cümle; hiçbirine dokunulmadı.

Bu davranış bir kullanıcı testinden çıktı ([issue #2](https://github.com/oguzhanozfe/tr-duzelt/issues/2)): ilk sürüm benzer bir girdide `orma na` parçalarını "Orda Na" yapıyor, `here we go again` içindeki kelimelere büyük harf koyuyordu.

## Neden bu yöntemler, neden ötekiler değil?

Her seçim ölçülerek yapıldı. Ölçüm verisi: gerçek insanların yazdığı, sonra elle düzeltilmiş 1.000 tweet (2.226 hata). Sütunların anlamı:

- **Düzeltilen:** hataların yüzde kaçı tam doğru düzeltildi (yüksek olsun).
- **Yanlış düzeltilen:** yüzde kaçı başka bir yanlışa çevrildi (düşük olsun).
- **Bozulan:** *doğru yazılmış* kelimelerin yüzde kaçı değiştirildi (en kötüsü bu; düşük olsun).

| Yöntem | Düzeltilen | Yanlış düzeltilen | Bozulan |
|---|---|---|---|
| Zemberek yazım denetleyicisi (bağlama bakmadan ilk öneri) | %33,5 | %28,6 | %1,47 |
| SymSpell (kelime listesi + harf mesafesi) | %42,0 | %45,1 | %11,82 |
| T5 yazım modeli (142 milyon parametre, hazır) | %19,2 | %8,2 | %1,27 |
| Zemberek'in hazır normalleştiricisi | %75,0 | %11,6 | %2,44 |
| tr-duzelt, ilk sürüm | %62,8 | %10,7 | %1,39 |
| tr-duzelt, elle ayarlı kurallar (sıralayıcısız) | %67,3 | %5,9 | %0,45 |
| tr-duzelt, öğrenilmiş sıralayıcı (25 Kasım) | %73,7 | %4,1 | %0,42 |
| **tr-duzelt, şimdiki: + kural son geçişi, yeni adaylar, eşik 0,7** | **%72,7** | **%2,7** | **%0,27** |
| **tr-duzelt `--bert`: belirsiz kelimede BERTurk'e de sorar** | **%75,7** | **%2,7** | **%0,35** |

Tek tek gerekçeler:

**Neden hazır bir yazım denetleyici yetmiyor?** Bağlama bakmayan denetleyici tanımadığı her kelimeyi en yakın kelimeye çevirir. Tabloda görüldüğü gibi düzelttiği kadar da yanlış düzeltir. Ayrıca yabancı kelimeleri ve özel adları da "düzeltir".

**Neden kelime listesi (SymSpell türü) değil?** Hızlıdır ama listesinde olmayan her şeyi yanlış sayar. Türkçede ekler yüzünden liste hiçbir zaman tam olmaz; yabancı kelimeleri de en yakın Türkçe kelimeye çevirir. (Denediğimiz SymSpell, Zemberek'in 300 bin kelimelik dağarcığından kurulmuş bir sözlükle ve varsayılan ayarlarla çalıştı; daha özenli bir sözlükle sonuç iyileşebilir ama sorunun kaynağı aynı kalır.)

**Neden Zemberek'in hazır normalleştiricisi değil?** En yakın rakip bu. Hataların aşağı yukarı aynı kadarını düzeltiyor ama doğru yazılmış her 40 kelimeden birini bozuyor ve yanlış düzeltmesi üç kat fazla. "Emin değilse dokunma" ilkesi onda yok.

**Neden eşikleri elle ayarlamak yerine öğrenilmiş sıralayıcı?** İlk sürümde "şu kadar benziyorsa, şu kadar sıksa düzelt" türünden kuralları elle yazmıştık ("elle ayarlı kurallar" satırı). Aynı adayları öğrenilmiş modele verince hem daha çok hata düzeldi hem daha az yanlış yapıldı. Üstelik elle kullanılamayan gürültülü kaynaklar (Zemberek'in 470 bin satırlık tablosu) ancak onunla işe yaradı: tabloyu elle kuralla kullandığımızda 32 doğru düzeltmeye karşı 74 yanlış getirmişti.

**Neden Türkçe karakterleri baştan topluca geri koymuyoruz?** Zemberek'te bunu yapan hazır bir araç var (Deasciifier). Tümüyle Türkçe karaktersiz yazılmış 364 tweette denedik: tek başına kullanınca doğru kelimelerin %2'sini bozuyor (`go → gö`). Şimdi yalnızca sıralayıcıya "ikinci görüş" olarak danışılıyor; son kararı sıralayıcı veriyor.

**Neden hazır bir sinir ağı (T5) değil?** Hugging Face'te Türkçe yazım için eğitilmiş 142 milyon parametreli bir T5 var ([cguzel/trspell-t5-efficient-small](https://huggingface.co/cguzel/trspell-t5-efficient-small)). Aynı tweetlerde ölçtük (tablodaki satır): hataların yalnızca %19'unu düzeltiyor, doğru kelimelerin %1,3'ünü bozuyor. Tweet dilini (`bugun`, `cok`, `dicem`) tanımıyor; temiz metinde eğitilmiş. Onu ayrı bir "aday kaynağı" olarak eklemenin de üst sınırını ölçtük: bizim düzelticinin kaçırdığı kelimelerin yalnızca %15'ine doğru öneri getiriyor (geliştirme bölümünde 324 kelimeden 48'i), üstüne 600 MB model ve ekran kartı/uzun süre istiyor. Değmedi. Kendiniz ölçmek için:

```bash
pip install torch transformers sentencepiece
python olcum/olc.py --sistem t5 --set gercek-test
```

Kelimenin ne demek istediğini anlamak için daha küçük ve daha uygun bir sinir ağı işe yaradı: [BERTurk](https://github.com/stefan-it/turkish-bert). İsteğe bağlı; aşağıda ["BERT kipi"](#bert-kipi-isteğe-bağlı).

**Neden LLM değil?** Büyük bir sohbet modeli bu işi büyük olasılıkla daha iyi yapar. Ama her metin için para ya da güçlü bir makine ister, metni dışarı gönderir ve bazen kimsenin istemediği yerleri de "güzelleştirir". Bu projede LLM yalnızca yardımcıdır: sonucu denetler ve kural önerir ([aşağıda](#kural-eklemek-ve-llmin-rolü)).

## Ne kadar iyi?

Yukarıdaki tablo gerçek veriydi. Bir de temiz cümleleri **bilerek bozup** geri düzeltmeyi deniyoruz; böylece her hata türünü ayrı ayrı görebiliyoruz (464 cümle, UD Turkish-IMST test bölümü):

| Test (düzeltilen / yanlış düzeltilen) | İlk sürüm | Zemberek normalleştirici | tr-duzelt (şimdiki) |
|---|---|---|---|
| Cümlede tek yazım hatası | %90 / %10 | %75 / %23 | **%84 / %2** |
| Cümlede üç yazım hatası | %91 / %8 | %74 / %23 | **%85 / %3** |
| Ortasından bölünmüş kelime (`orma na`) | %0 / hepsi | %70 / %28 | **%88 / %2** |
| Yoğun gürültü: hepsi bir arada + araya İngilizce | %74 / %21 | %71 / %23 | **%85 / %2** |
| Araya giren İngilizce: bozulan kelime | %1,50 | %3,08 | **%0,11** |
| Temiz metin: bozulan kelime | %0,32 | %1,32 | **%0,12** |

Okurken bilinmesi gerekenler:

- Bu sayılar yalnızca **2. durağı** (kelime düzeltme) ölçer; noktalama ayrı ölçülür (aşağıda).
- Gerçek tweet verisinin ilk 1.000'i ayar yapmak için kullanıldı, sonuçlar hiç bakılmamış son 1.000'den. Sıralayıcı tweetlerle eğitilmedi.
- Yapay testlerdeki bozma yöntemi, sıralayıcının eğitimindekiyle aynı aileden (cümleler farklı). Bu yüzden yapay sonuçlar gerçek hayattan iyimserdir; asıl gösterge gerçek tweet tablosudur.
- Tweet verisini düzeltenlerin de kendi tercihleri var. "Bozulan" sayılanların bir kısmı aslında doğru düzeltme (`lüften → lütfen`, `unlu → ünlü`).

Kendiniz ölçmek için (veriler ilk çalıştırmada indirilir):

```bash
python olcum/olc.py                                   # bu depo, bütün setler
python olcum/olc.py --sistem zemberek-normalizer      # başka bir yöntem
python olcum/olc.py --set gercek-test --ornek 5       # hatalı örnekleri de göster
python olcum/hata_turleri.py                          # hangi tür hataların kaçı düzeliyor
```

**Noktalama ve büyük harf** (F1: 1 = kusursuz; eğitimde hiç görülmemiş belgeler):

| Test kümesi | Nokta | Virgül | Soru işareti | Büyük harf |
|---|---|---|---|---|
| Gazete (21,6 bin kelime) | 0,806 | 0,332 | 0,580 | 0,784 |
| Roman (13,2 bin kelime) | 0,760 | 0,194 | 0,580 | 0,584 |
| Öğrenci yazıları, alan dışı (88 bin kelime) | 0,660 | 0,218 | 0,507 | – |

Bu tablo eğitim betiğinin kendi ayırdığı belgelerden. Her yöntemi **aynı** veride ölçmek için ayrı bir betik var: `python olcum/noktalama_olc.py`. Verisi UD Turkish-IMST'nin geliştirme ve test bölümleri (16 bin kelime); bu veriyle hiçbir model eğitilmedi. Sonuçlar (F1, test bölümü):

| Yöntem | Nokta | Virgül | Soru işareti |
|---|---|---|---|
| Bizim CRF (düzelticinin kullandığı ayarlarla) | 0,675 | 0,079 | 0,341 |
| Hazır BERT modeli ([uygarkurt/bert-restore-punctuation-turkish](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish)) | 0,276 | 0,300 | 0,171 |
| **Bizim BERTurk etiketleyicimiz (`--bert`)** | **0,782** | **0,433** | **0,463** |

Hazır BERT modeli kendi model kartında nokta için 0,96 bildiriyor ama burada cümle sonlarının çoğuna nokta yerine virgül koyuyor (`güzeldi, dışarı çıktık, sonra eve döndük,`). Bu haliyle kullanılamaz. Bizim CRF'in virgülü düşük çünkü düzeltici onu bilerek çok temkinli çalıştırıyor: yalnızca çok emin olduğu yere virgül koyuyor. Kendi BERTurk etiketleyicimiz CRF'le aynı veriyle eğitildi ve üç işarette de ondan iyi; koyduğu virgüllerin %75'i doğru (CRF'te %69).

## BERT kipi (isteğe bağlı)

`--bert` iki şeyi değiştirir. İkisi de [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) üzerine kurulu. BERTurk, milyarlarca kelimelik Türkçe metinle eğitilmiş, cümlenin tamamına bakan bir dil modeli (110 milyon parametre). Sohbet modeli değil: metin üretmez, yalnızca puan verir.

1. **Kelime seçimi.** Varsayılan sıralayıcı yalnızca yandaki kelimelere bakabiliyor. `olum` mu `ölüm` mü `oğlum` mu, `bende` mi `ben de` mi? Bunu çoğu zaman cümlenin geri kalanı söyler. BERT kipinde, taban sıralayıcının **emin olamadığı** kelimelerde her aday cümlenin içine konup BERTurk'e sorulur: "Bu cümlede bu kelime ne kadar olası?" Bu puan sıralayıcıya beş yeni ipucu olarak girer. Sıralayıcı bu ipuçlarıyla yeniden eğitildi (`siralayici_bert.json`). Emin olunan kelimelere BERT hiç sorulmaz; işlemcide süreyi üç kat kısaltan şey bu.
2. **Noktalama.** CRF'in yerine BERTurk üzerine eğittiğimiz noktalama etiketleyicisi (`noktalama_bert/`). Büyük harfi yine CRF verir. Virgülü ancak %70'ten, soru işaretini %60'tan eminse koyar.

Sonuç (gerçek tweetler, test):

| | Düzeltilen | Yanlış düzeltilen | Bozulan |
|---|---|---|---|
| Varsayılan | %72,7 | %2,7 | %0,27 |
| `--bert` | %75,7 | %2,7 | %0,35 |
| `--temkinli` (eşik 0,9) | %66,0 | %1,9 | %0,12 |
| `--bert --temkinli` | %68,9 | %1,8 | %0,13 |

`--bert` aynı yanlış oranında daha çok düzeltir. Test tweetlerinde bozulan doğru kelime biraz daha yüksek (%0,35'e karşı %0,27), geliştirme tweetlerinde ise daha düşük (%0,26'ya karşı %0,36). Fark ölçüm gürültüsü kadar. Yapay testlerde düzeltmedeki fark daha açık: cümlede tek yazım hatası %84'ten %89'a, üç yazım hatası %85'ten %90'a.

**Ekran kartı gerekmez.** İşlemcide kelime seçimindeki BERTurk otomatik olarak 8 bitlik tam sayı hesabına geçer: 1,7 kat hızlı, sonuç aynı (geliştirme tweetlerinde %74,8 / %2,9 / %0,28). Tek çekirdekte kelime seçimi saniyede ~65 kelime (aynı ölçümde BERT'siz ~440), noktalama ~300 kelime. Ekran kartı varsa kendiliğinden kullanılır.

Kurulum:

```bash
pip install torch transformers          # yalnızca CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python tr_duzelt.py --bert "olum sen ordayim diyosun ama gelmedin"
```

İlk çalıştırmada BERTurk (~440 MB) Hugging Face'ten iner. Noktalama modeli (`noktalama_bert/`, 211 MB) depoda değil, [sürümler sayfasında](https://github.com/oguzhanozfe/tr-duzelt/releases/tag/noktalama-bert-v1). `noktalama_bert.zip` dosyasını indirip `tr_duzelt.py`'nin bulunduğu klasöre açın:

```powershell
Invoke-WebRequest https://github.com/oguzhanozfe/tr-duzelt/releases/download/noktalama-bert-v1/noktalama_bert.zip -OutFile noktalama_bert.zip
Expand-Archive noktalama_bert.zip -DestinationPath .
```

Klasör yoksa `--bert` noktalamayı CRF'le yapar, kelime seçimi yine BERT'li çalışır. Modeli kendiniz de eğitebilirsiniz (ekran kartıyla birkaç dakika; aşağıda).

## Yapamadıkları

- **Yazımı doğru, anlamı yanlış kelimeler.** `yaşam küçük` (doğrusu `yaşım küçük`), `oldum/öldüm` gibi durumlarda iki kelime de geçerli Türkçedir. Doğrusunu seçmek cümleyi anlamayı gerektirir; varsayılan kip yalnızca yandaki kelimelere bakabiliyor. `--bert` bazılarını yakalar; emin olmadığı için çoğunu bırakır.
- **de/da ve ki.** `bende → ben de` artık bazen düzeliyor, ama güvenilir değil: gerçek tweetlerde bitişik yazılmış kelime hatalarının yaklaşık yarısı düzeltiliyor.
- **Birden çok harfi yanlış kelimeler.** `dokmusa` (dolmuşa) gibi iki harfi birden yanlış kelimeler için aday üretilemiyor; kelime korunur.
- **Virgül.** Model yalnızca çok emin olduğunda virgül koyar; eksik virgül çok kalır.
- **Devrik cümle, konuşma, liste.** Noktalama modeli düz anlatımda iyi, bunlarda zayıf. `Aslan kaçtı ormana` cümlesini `Aslan kaçtı. Ormana…` diye bölebilir.
- **Paragraf.** Denenen yöntem rastgele bölmekten iyi çıkmadı; kodda duruyor ama kapalı (`--paragraf`).
- **Hız.** `zemberek-python` saf Python olduğu için yavaş: açılış yaklaşık 20 saniye, sonra tek süreçte saniyede kabaca 25-35 kelime (bütün aşamalar). Büyük dökümler için `--toplu` birden çok süreç kullanır; süreler [Kullanım](#kullanım) bölümünde.

## Kural eklemek ve LLM'in rolü

Düzelticinin bilmediği, tekrar eden bir yanlış gördüğünüzde `tr_duzelt.py` içindeki üç listeden birine bir satır eklemeniz yeter:

| Liste | Ne zaman | Örnek satır |
|---|---|---|
| `TAM_DUZELT` | Kelimenin tamamı eşleşmeli (kısaltmalar, bitişik bağlaçlar) | `'gbi': 'gibi'`, `'yada': 'ya da'` |
| `GOVDE_DUZELT` | Kelimenin başı eşleşmeli; ekler korunur | `'herkez': 'herkes'` → `herkeze` de `herkese` olur |
| `KONUSMA_GECERLI` | Geçerli ama konuşma diline ait biçimler | `'burda': 'burada'` |

Sözlükten gelen düzeltme her zaman uygulanır. Bu yüzden yalnızca **her bağlamda** doğru olan kuralları ekleyin (`direk → direkt` eklemeyin: "direk" geçerli bir kelime).

**LLM burada ne yapıyor?** Metni düzeltmiyor. İsteğe bağlı `--eval`, girdiyle çıktıyı Gemini'ye gönderir; o da şüpheli yerleri ve olası kuralları listeler. Öneriyi siz okur, doğruysa listeye eklersiniz. Yani LLM bir kere, kural yazarken yardım eder; kural eklendikten sonra düzeltme yine yerelde ve bedavaya çalışır. Eklediğiniz kuralın bir şeyi bozmadığını `python olcum/olc.py` ile görebilirsiniz.

`--eval` metni Google'a gönderir: **kişisel, gizli ya da hassas metinlerde kullanmayın.** Ayrıntı [Kullanım](#kullanım) bölümünde.

## Kurulum

Python 3.11 ve 3.12 ile denendi. `zemberek-python`, Python 3.13+ ve NumPy 2 ile çalışmıyor.

```bash
pip install -r requirements.txt
```

### Windows (yönetici hakkı olmadan)

Bu yol hiçbir adımda yönetici onayı istemez: her şey kullanıcı klasörünüze kurulur, sistemdeki başka bir Python'a dokunulmaz ve C derleyicisi gerekmez.

> Düzelticinin Windows'ta (PowerShell, Python 3.11.9) çalıştığı bir kullanıcı testiyle doğrulandı. Aşağıdaki uv adımlarının kendisi Linux'ta denendi. Takıldığınız yer olursa issue açın.

**1. Depoyu indirin.** [ZIP olarak indirip](https://github.com/oguzhanozfe/tr-duzelt/archive/HEAD.zip) bir klasöre açın. Git gerekmez.

**2. uv'yi kurun.** [uv](https://docs.astral.sh/uv/), doğru Python sürümünü kendisi indirip yöneten bir araçtır. Başlat menüsünden PowerShell'i normal şekilde açın ("Yönetici olarak çalıştır" değil) ve şunu çalıştırın:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Bitince pencereyi kapatıp yeni bir PowerShell açın; `uv --version` sürüm yazmalı.

**3. Ortamı kurun.** Zip'i açtığınız, içinde `tr_duzelt.py` bulunan klasöre girin:

```powershell
cd $HOME\Downloads\tr-duzelt-main      # klasörün adı sizde farklıysa (ör. tr-duzelt-master) onu yazın
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

python tr_duzelt.py --yolculuk "..."                       # adım adım: hangi kelimeye neden dokunuldu
python tr_duzelt.py --dosya girdi.txt --ayrinti            # yapılan kelime düzeltmelerini de listeler
python tr_duzelt.py --dosya girdi.txt --cikti cikti.txt    # sonucu UTF-8 dosyaya yazar
python tr_duzelt.py --temkinli "..."                        # yalnızca çok emin olduğunu düzeltir (yanlış ~%2)
python tr_duzelt.py --bert "..."                            # cümlenin tamamına bakar (bkz. BERT kipi)
```

**Büyük dökümler (100 bin – 200 bin metin).** Her satırı ayrı bir metin sayar, çıktıya aynı sırayla satır satır yazar:

```bash
python tr_duzelt.py --toplu dokum.txt --cikti duzeltilmis.txt              # varsayılan: çekirdek - 1 süreç, en çok 8
python tr_duzelt.py --toplu dokum.txt --cikti duzeltilmis.txt --isci 3     # dizüstü: 3 süreç
python tr_duzelt.py --toplu dokum.txt --cikti duzeltilmis.txt --bert       # BERT kipi
```

- Girdideki her satıra çıktıda tam bir satır karşılık gelir. Bir satırda beklenmedik bir hata olursa o satır olduğu gibi yazılır, iş durmaz.
- Yarıda kalırsa (elektrik, Ctrl+C) aynı komutu yeniden çalıştırın: çıktıda bitmiş satırlar atlanır, kalan yerden devam eder.
- Her süreç varsayılan kipte ~1 GB, `--bert` ile ~2,5 GB bellek tutar. İlerleme ve tahmini kalan süre ekrana yazılır.

Ölçülen süreler. Girdi: 2.000 gerçek tweet, satır başına ortalama 8,5 kelime. İlk geçiş, önbellek soğukken. 32 çekirdekli bir masaüstü, ekran kartı kapalı:

| Kip | Süreç | Saniyede satır | 100 bin metin | 200 bin metin |
|---|---|---|---|---|
| Varsayılan | 8 | 24 | ~70 dk | ~2,3 saat |
| Varsayılan | 3 (dizüstü gibi) | 11 | ~2,5 saat | ~5 saat |
| `--bert`, yalnızca işlemci | 8 | 17 | ~1,6 saat | ~3,3 saat |

Metinleriniz daha uzunsa süre kelime sayısıyla orantılı uzar (8 süreçle saniyede ~200 kelime).

Kendi yazdığımız 20 düzeltilmemiş metin ve iki kipin çıktısı `ornekler/` klasöründe (`girdi.txt`, `cikti.txt`, `cikti_bert.txt`). Bu metinler bu sürümde eklenen kuralları (`olum`, `ordayim`, `hersey`, `giriyorm`, `bi kac`) göstermek için yazıldı; asıl kanıt yukarıdaki gerçek tweet testi. `cikti_bert.txt` depoda olmayan `noktalama_bert/` modeliyle üretildi. Bir kaçı:

| Girdi | Varsayılan | `--bert` |
|---|---|---|
| herkez bu filmi izlemis bende izlemek istiyorum ama vaktim yok | Herkes bu filmi izlemiş bende izlemek istiyorum ama vaktim yok. | Herkes bu filmi izlemiş. Ben de izlemek istiyorum ama vaktim yok. |
| hocam odevi yarina kadar teslim edebilirmiyiz cok yogunuz bu hafta | Hocam ödevi yarına kadar teslim edebilir miyiz? Çok yoğunuz. Bu hafta. | Hocam ödevi yarına kadar teslim edebilir miyiz? Çok yoğunuz bu hafta. |
| yarin hava nasil olcak bilen varmi yagmur yagicak mi | Yarın hava nasıl olacak bilen var mı? Yağmur yagicak mi? | Yarın hava nasıl olacak? Bilen var mı? Yağmur yagicak mı? |
| giriyorm simdi toplantiya sonra konusuruz tmm mı | Giriyorum şimdi. Toplantıya sonra konuşuruz tamam mı? | Giriyorum şimdi toplantıya. Sonra konuşuruz tamam mı? |

Son iki satır kalan zayıflıkları da gösteriyor: `yagicak` (yağacak) gibi gelecek zamanın konuşma biçimleri her zaman düzelmiyor.

**Ne kadar cesur olsun?** `TR_DUZELT_ESIK` ortam değişkeni sıralayıcının eşiğini değiştirir (varsayılan 0,7; geliştirme tweetlerinde seçildi). Yükseltirseniz daha az ama daha emin düzeltir. Gerçek tweetlerin test bölümünde:

| Eşik | Düzeltilen | Yanlış düzeltilen | Bozulan |
|---|---|---|---|
| 0,5 | %74,4 | %3,6 | %0,43 |
| 0,7 (varsayılan) | %72,7 | %2,7 | %0,27 |
| 0,9 | %66,0 | %1,9 | %0,12 |

```powershell
$env:TR_DUZELT_ESIK = "0.9"      # PowerShell;  Linux/macOS: TR_DUZELT_ESIK=0.9 python tr_duzelt.py ...
```

**İsteğe bağlı Gemini denetimi:**

```powershell
# PowerShell'de yalnızca bu oturum için API anahtarı tanımla, sonra eval çalıştır
$env:GEMINI_API_KEY = "<Google AI Studio API key>"
python tr_duzelt.py --dosya girdi.txt --eval
```

`--eval` varsayılan olarak `gemini-4-hydrogen` kullanır. Bu seçenek verilmedikçe ağ isteği yapılmaz. Eval, hem girdinin tamamını hem düzeltilmiş çıktıyı Google Gemini API'ye gönderir. Google'ın ücretsiz kullanım koşullarında gönderilen içeriklerin ürünleri geliştirmek için kullanılabileceği ve insan incelemesine konu olabileceği belirtiliyor; **kişisel, gizli veya hassas metinleri göndermeyin**. Ücretsiz eval'i yalnızca hassas olmayan geliştirme/test girdilerinde kullanın ve kullanım koşullarını kontrol edin. Bkz. [Gemini API fiyatlandırma/veri kullanımı](https://ai.google.dev/gemini-api/docs/pricing) ve [Gemini API şartları](https://ai.google.dev/gemini-api/docs/terms).

Gemini yalnızca olası sorunları ve kural önerilerini raporlar; metni veya kural listelerini otomatik değiştirmez. Bir öneriyi ancak amaçlanan doğru metni elle doğruladıktan sonra ekleyin. Belirsiz/gürültülü girdilerde eval de hata yapabilir; sonucu kesin doğruluk ölçütü saymayın.

**Python içinden:**

```python
from tr_duzelt import Duzeltici

d = Duzeltici("noktalama.crfsuite")      # yükleme ~20 sn (dil modeli + morfoloji + tablolar)
print(d.duzelt("herkez geldimi bilmiyorum"))
```

## Dosyalar: ne nerede

| Dosya | İçinde ne var |
|---|---|
| `tr_duzelt.py` | Düzelticinin tamamı: aday üretimi, sıralayıcıyı çalıştırma, noktalama, büyük harf, komut satırı |
| `siralayici.json` | Öğrenilmiş aday sıralayıcı (1,4 MB). Silerseniz düzeltici elle ayarlı kurallara döner |
| `noktalama.crfsuite` | Öğrenilmiş noktalama + büyük harf modeli (5,6 MB) |
| `egit_siralayici.py` | Sıralayıcıyı eğitir (cümleleri bozar, adayları çıkarır, modeli kurar) |
| `egit_noktalama.py` | Noktalama modelini eğitir |
| `siralayici_bert.json` | `--bert` kipinin sıralayıcısı: aynı 50 ipucu + BERTurk'ün 5 ipucu (1,3 MB) |
| `egit_noktalama_bert.py` | `--bert` kipinin noktalama modelini (`noktalama_bert/`) eğitir |
| `olcum/` | Ölçüm düzeneği: `veri.py` test setlerini kurar, `olcut.py` sayar, `sistemler.py` karşılaştırılan yöntemler, `olc.py` çalıştırır; `hata_turleri.py` hata türüne göre döküm verir, `esik_tara.py` sıralayıcının eşiğini tarar |
| `tr_eval.py`, `tests/` | İsteğe bağlı Gemini denetimi ve testleri (kural geçişi ve BERT kararı için de) |
| `ornekler/` | Kendi yazdığımız 20 düzeltilmemiş metin ve iki kipin çıktısı |
| `CLAUDE.md` | Projeyi devralacak geliştirici (ya da Claude Code) için notlar: kurallar, tuzaklar, sıradaki işler |

## Modelleri yeniden eğitmek

**Sıralayıcı** (kural ya da aday üretimi değişince yeniden eğitmek gerekir; yaklaşık 1 saat sürer):

```bash
pip install lightgbm            # yalnızca eğitim için; düzelticinin kendisi istemez
python egit_siralayici.py       # siralayici.json dosyasını yeniden yazar
python olcum/olc.py             # sonucu ölç
```

**Noktalama modeli:**

```bash
python egit_noktalama.py --trmor2018 trmor2018.train --ud *.conllu \
    --cikti noktalama.crfsuite --test-kitap 14 17
```

**BERT kipi:** sıralayıcı için `python egit_siralayici.py --bert` (tablo üretimi BERTurk'ü ekran kartında çalıştırır; tabloyu ayrı süreçlere bölmek için CLAUDE.md'deki komutlar). Noktalama için, yukarıdaki verilerle:

```bash
pip install torch transformers
python egit_noktalama_bert.py --trmor2018 trmor2018.train --ud *.conllu --cikti noktalama_bert --test-kitap 14 17
```

RTX 3090'da 3 tur yaklaşık 3 dakika sürdü (verinin okunması hariç). Ayrılmış TrMor belgelerinde nokta 0,880, virgül 0,567, soru 0,771 F1 (CRF: 0,806 / 0,332 / 0,580).

Kendi alanınızdan düzgün yazılmış metinle eğitmek hem başarımı artırır hem lisans sorusunu sadeleştirir. Ayrıntılar `egit_noktalama.py` başlığında. Windows'ta eğitirken `--cikti` yolunda Türkçe karakter olmasın: CRFsuite model dosyasını C kütüphanesiyle yazar ve bu tür yolları açamaz. (`tr_duzelt.py` modeli Python üzerinden okuduğu için bundan etkilenmez.)

## Veri, araçlar ve atıflar

**Modellerin eğitildiği veriler:**

- `noktalama.crfsuite` (≈858 bin kelime): [TrMor2018](https://github.com/ai-ku/TrMor2018) — Dayanık, Akyürek, Yüret (2018), MIT; Universal Dependencies Türkçe ağaç yapıları: [BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN), [Kenet](https://github.com/UniversalDependencies/UD_Turkish-Kenet), [Penn](https://github.com/UniversalDependencies/UD_Turkish-Penn), [Tourism](https://github.com/UniversalDependencies/UD_Turkish-Tourism), [FrameNet](https://github.com/UniversalDependencies/UD_Turkish-FrameNet), [GB](https://github.com/UniversalDependencies/UD_Turkish-GB) (CC BY-SA 4.0), [PUD](https://github.com/UniversalDependencies/UD_Turkish-PUD) (CC BY-SA 3.0)
- `siralayici.json`, `siralayici_bert.json`: UD Turkish-BOUN eğitim bölümünün yapay olarak bozulmuş cümleleri (CC BY-SA 4.0)
- `noktalama_bert/`: `noktalama.crfsuite` ile aynı veriler; temel model [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) (Stefan Schweter, MDZ Digital Library; MIT)

**Model keşfi ve araştırma:** Esra Bingöl.

Model dosyaları bu derlemlerden öğrenilmiş istatistikleri içerir; yeniden dağıtırken ya da ticari bir üründe kullanırken bu lisansların atıf ve aynı lisansla paylaşma koşullarını dikkate alın.

**Yalnızca ölçüm için kullanılan, eğitime girmeyen ve bu depoda bulunmayan veriler:**

- [#Turki$hTweets](https://github.com/atubakoksal/annotated_tweets) — Koksal, Bozal, Yürekli, Gezici (2020). *#Turki$hTweets: A Benchmark Dataset for Turkish Text Correction.* Findings of EMNLP 2020. Gerçek kullanıcı hataları, elle düzeltilmiş 2.000 tweet.
- [UD Turkish-IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST) (CC BY-NC-SA 4.0) — yapay bozma testleri
- Bilkent Turkish Writings Dataset — Yilmaz, S. F. (2025). *Compilation of Bilkent Turkish Writings Dataset*. Zenodo. https://doi.org/10.5281/zenodo.15498155. Özgün kaynak: [Bilkent Üniversitesi Türkçe Birimi](https://stars.bilkent.edu.tr/turkce/). Öğrenci yazısı ve paragraf ölçümleri.

**Kütüphaneler:** [Zemberek-NLP](https://github.com/ahmetaa/zemberek-nlp) (Akın & Akın) ve Python uyarlaması [zemberek-python](https://github.com/Loodos/zemberek-python) — kelime çözümleme, yazım önerileri, dil modeli, hazır tablolar, Deasciifier; [python-crfsuite](https://github.com/scrapinghub/python-crfsuite); eğitimde [LightGBM](https://github.com/microsoft/LightGBM).

## Dayandığı çalışmalar

- Torunoğlu, D. & Eryiğit, G. (2014). A Cascaded Approach for Social Media Text Normalization of Turkish. *LASM @ EACL 2014*, 62–70. — kademeli normalizasyon mimarisi
- Kernighan, M., Church, K. & Gale, W. (1990). A Spelling Correction Program Based on a Noisy Channel Model. *COLING*. — aday + dil modeli yaklaşımı
- Mays, E., Damerau, F. & Mercer, R. (1991). Context Based Spelling Correction. *Information Processing & Management*. — gerçek-kelime hataları
- Stolcke, A. & Shriberg, E. (1996). Automatic Linguistic Segmentation of Conversational Speech. *ICSLP*. — noktalama için dil modeli özniteliği
- Lafferty, J., McCallum, A. & Pereira, F. (2001). Conditional Random Fields. *ICML*.
- Ke, G. vd. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*. — sıralayıcının yöntemi
- Hearst, M. (1997). TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages. *Computational Linguistics* 23(1).
