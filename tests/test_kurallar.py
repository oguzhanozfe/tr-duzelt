import unittest
from types import SimpleNamespace

from tr_duzelt import YazimDuzeltici


class KuralSonGecisTests(unittest.TestCase):
    """Seçilen düzeltme tablodaki bir yanlış biçimse o da düzeltilir (Zemberek yüklemeden, sahte gecerli ile)."""

    def gecis(self, kelimeler, sonuc):
        sahte = SimpleNamespace(gecerli=lambda w: True)
        sonuc = list(sonuc)
        YazimDuzeltici._kural_son_gecis(sahte, kelimeler, sonuc)
        return sonuc

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


if __name__ == '__main__':
    unittest.main()
