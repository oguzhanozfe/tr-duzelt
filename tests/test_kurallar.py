import unittest
from types import SimpleNamespace

from tr_duzelt import YazimDuzeltici, _uyumlu_soru_eki, _gelecek_zaman, _unlu_yerlestir


class KuralSonGecisTests(unittest.TestCase):
    """Seçilen düzeltme tablodaki bir yanlış biçimse o da düzeltilir (Zemberek yüklemeden, sahte gecerli ile)."""

    def gecis(self, kelimeler, sonuc):
        gecersiz = {'varmı', 'hanımmi'}
        sahte = SimpleNamespace(gecerli=lambda w: True, duzgun=lambda w: w not in gecersiz)
        sonuc = list(sonuc)
        YazimDuzeltici._kural_son_gecis(sahte, kelimeler, sonuc)
        return sonuc

    def test_soru_eki_uyumu(self):
        self.assertEqual(_uyumlu_soru_eki('hanım', 'mi'), 'mı')
        self.assertEqual(_uyumlu_soru_eki('gül', 'musun'), 'müsün')
        self.assertEqual(_uyumlu_soru_eki('geldin', 'mı'), 'mi')

    def test_bitisik_soru_eki_ayrilir(self):
        self.assertEqual(self.gecis(['varmi'], [(('varmı',), 'aksan')]), [(('var', 'mı'), 'sözlük')])

    def test_konusma_govdesi(self):
        self.assertEqual(self.gecis(['ordayim'], [(('ordayım',), 'aksan')]), [(('oradayım',), 'sözlük')])

    def test_hicbir_yalniz_duzeltilmisse_birlesir(self):
        self.assertEqual(self.gecis(['hic', 'bi'], [(('hiç',), 'aksan'), (('bir',), 'sözlük')]),
                         [((), 'birleştirme'), (('hiçbir',), 'birleştirme')])
        self.assertEqual(self.gecis(['hiç', 'bir'], [(('hiç',), 'aynı'), (('bir',), 'aynı')]),
                         [(('hiç',), 'aynı'), (('bir',), 'aynı')])

    def test_aksan_secimi_tablodan_gecer(self):
        self.assertEqual(self.gecis(['hersey'], [(('herşey',), 'aksan')]), [(('her', 'şey'), 'sözlük')])
        self.assertEqual(self.gecis(['birseydir'], [(('birşeydir',), 'aksan')]), [(('bir', 'şeydir'), 'sözlük')])
        self.assertEqual(self.gecis(['bıde'], [(('bide',), 'aksan')]), [(('bir', 'de'), 'sözlük')])
        self.assertEqual(self.gecis(['sagolsun'], [(('sağolsun',), 'aksan')]), [(('sağ', 'olsun'), 'sözlük')])

    def test_dokunulmayan_kelimeye_karismaz(self):
        # kelime olduğu gibi bırakıldıysa (özel isim olabilir) son geçiş onu değiştirmez
        self.assertEqual(self.gecis(['birgün'], [(('birgün',), 'aynı')]), [(('birgün',), 'aynı')])
        self.assertEqual(self.gecis(['evde'], [(('evde',), 'aksan')]), [(('evde',), 'aksan')])

    def test_gecersiz_parca_cikarsa_birakir(self):
        sahte = SimpleNamespace(gecerli=lambda w: w != 'şey')
        sonuc = [(('herşey',), 'aksan')]
        YazimDuzeltici._kural_son_gecis(sahte, ['hersey'], sonuc)
        self.assertEqual(sonuc, [(('herşey',), 'aksan')])


class GelecekZamanTests(unittest.TestCase):
    def test_daralmis_gelecek_zaman(self):
        self.assertIn('bekleyecez', _gelecek_zaman('beklicez'))
        self.assertIn('başlayacak', _gelecek_zaman('başlicak'))
        self.assertIn('yagacak', _gelecek_zaman('yagicak'))
        self.assertIn('alacam', _gelecek_zaman('alcam'))
        self.assertIn('gidecektin', _gelecek_zaman('gidicektin'))

    def test_ilgisiz_kelimeye_bicim_uretmez(self):
        self.assertEqual(_gelecek_zaman('dicem'), [])      # gövde tek harf: tahmin yürütme
        self.assertEqual(_gelecek_zaman('kitap'), [])


class UnluYerlestirTests(unittest.TestCase):
    def test_dusmus_unlu_geri_konur(self):
        self.assertIn('senin', _unlu_yerlestir('snn'))
        self.assertIn('giriyorum', _unlu_yerlestir('giriyorm'))
        self.assertIn('kalırsın', _unlu_yerlestir('kalrsn'))

    def test_unlu_uyumu_ve_sinir(self):
        self.assertNotIn('giriyorüm', _unlu_yerlestir('giriyorm'))     # 'o' kalın: ince ünlü (ü) eklenmez
        self.assertTrue(all(len(b) <= len('kalrsn') + 2 for b in _unlu_yerlestir('kalrsn')))   # en çok iki ünlü


if __name__ == '__main__':
    unittest.main()
