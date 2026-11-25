#!/usr/bin/env python3
"""Sıralayıcının eşiğini gerçek tweetlerin GELİŞTİRME bölümünde tarar: eşik yükseldikçe ne kazanılıyor, ne kaybediliyor?

  python olcum/esik_tara.py                 # 0.3 0.4 0.5 0.6 0.7 0.8
  python olcum/esik_tara.py 0.5 0.9

Sıralayıcıyı yeniden eğittikten sonra çalıştırın; seçilen eşik egit_siralayici.py'nin yazdığı siralayici.json
içindeki "esik" alanına (ya da TR_DUZELT_ESIK ortam değişkenine) yazılır. Test bölümüne bakarak eşik seçmeyin.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import veri        # noqa: E402
import olcut       # noqa: E402
import sistemler   # noqa: E402


def main():
    esikler = [float(x) for x in sys.argv[1:]] or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    s = sistemler.TrDuzelt()
    if s.yazim.siralayici is None:
        raise SystemExit('siralayici.json bulunamadı: önce egit_siralayici.py çalıştırın')
    ciftler = [(g, d) for g, d, _ in veri.tweetler()[:1000]]
    # olasılıklar eşikten bağımsız: bir kez hesapla, her eşik için yalnızca kararı yeniden ver
    asil, bellek = s.yazim.siralayici.olasilik, {}

    def olasilik(satirlar):
        anahtar = tuple(tuple(x) for x in satirlar)
        if anahtar not in bellek:
            bellek[anahtar] = asil(satirlar)
        return bellek[anahtar]
    s.yazim.siralayici.olasilik = olasilik
    print('%5s | %10s %9s %11s | %8s | %6s %5s' % ('eşik', 'düzeltildi', 'bırakıldı', 'yanlış düz.', 'bozulan', 'isabet', 'F0.5'))
    for esik in esikler:
        s.yazim.siralayici.esik = esik
        r = olcut.degerlendir(ciftler, [s(g) for g, _ in ciftler])
        h = max(1, r['hata'])
        print('%5.2f | %9.1f%% %8.1f%% %10.1f%% | %7.2f%% | %5.1f%% %5.1f' % (
            esik, 100.0 * r['duzeltildi'] / h, 100.0 * r['birakildi'] / h, 100.0 * r['yanlis_duzeltildi'] / h,
            r['bozma'], r['isabet'], r['f05']), flush=True)


if __name__ == '__main__':
    main()
