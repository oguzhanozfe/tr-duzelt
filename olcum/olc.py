#!/usr/bin/env python3
"""Bir yöntemi sabit ölçüm setlerinde çalıştırır ve ne kadarını düzelttiğini / bozduğunu yazar.

  python olcum/olc.py                                   # bu depodaki düzeltici, bütün setler
  python olcum/olc.py --sistem zemberek-normalizer      # başka bir yöntem
  python olcum/olc.py --set gercek-test yapay-yogun     # yalnızca bazı setler

Sütunlar:
  düzeltildi   : hataların yüzde kaçı tam doğru düzeltildi
  bırakıldı    : yüzde kaçına dokunulmadı (hata olduğu gibi kaldı)
  yanlış düz.  : yüzde kaçı başka bir yanlışa çevrildi
  bozulan      : DOĞRU yazılmış kelimelerin yüzde kaçı değiştirildi (en kötü hata türü)
  isabet       : yapılan değişikliklerin yüzde kaçı doğruydu
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import veri        # noqa: E402
import olcut       # noqa: E402
import sistemler   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sistem', default='tr-duzelt',
                    help='hicbir-sey | tr-duzelt | zemberek-normalizer | zemberek-yazim | symspell')
    ap.add_argument('--kod', help='tr-duzelt için başka bir sürümün klasörü (eski sürümle karşılaştırmak için)')
    ap.add_argument('--set', nargs='*', help='ölçülecek setler (varsayılan: hepsi)')
    ap.add_argument('--cikti', help='sonuçların yazılacağı JSON dosyası')
    ap.add_argument('--ornek', type=int, default=0, help='her setten bu kadar hatalı örnek göster')
    a = ap.parse_args()

    sistem = sistemler.kur(a.sistem, a.kod)
    gecerli = sistem.gecerli if hasattr(sistem, 'gecerli') else sistemler.TrDuzelt().gecerli
    setler = veri.butun_setler(gecerli)
    print('sistem: %s' % sistem.ad)
    print('%-28s %6s %6s | %10s %9s %11s | %8s | %6s %5s' % (
        'set', 'cümle', 'hata', 'düzeltildi', 'bırakıldı', 'yanlış düz.', 'bozulan', 'isabet', 'F0.5'))
    sonuc = {'sistem': sistem.ad}
    for ad, ciftler in setler.items():
        if a.set and ad not in a.set:
            continue
        t = time.time()
        ciktilar = [sistem(g) for g, _ in ciftler]
        s = olcut.degerlendir(ciftler, ciktilar)
        s['sure'] = time.time() - t
        sonuc[ad] = s
        h = max(1, s['hata'])
        print('%-28s %6d %6d | %9.1f%% %8.1f%% %10.1f%% | %7.2f%% | %5.1f%% %5.1f' % (
            ad, len(ciftler), s['hata'], 100.0 * s['duzeltildi'] / h, 100.0 * s['birakildi'] / h,
            100.0 * s['yanlis_duzeltildi'] / h, s['bozma'], s['isabet'], s['f05']), flush=True)
        gosterilen = 0
        for (g, d), c in zip(ciftler, ciktilar):
            if gosterilen >= a.ornek:
                break
            if c != d:
                gosterilen += 1
                print('    girdi : %s\n    çıktı : %s\n    doğru : %s' % (' '.join(g), ' '.join(c), ' '.join(d)))
    if a.cikti:
        with open(a.cikti, 'w', encoding='utf-8') as f:
            json.dump(sonuc, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
