import unittest
from types import SimpleNamespace

from tr_duzelt import BertNoktalayici


class BertNoktalamaKarariTests(unittest.TestCase):
    """Emin değilse dokunma: eşiği geçmeyen virgül konmaz, eşiği geçmeyen soru işareti noktaya döner (model yüklemeden)."""

    def karar(self, p):
        return BertNoktalayici.karar(SimpleNamespace(virgul_esik=0.7, soru_esik=0.6), p)

    def test_emin_virgul_konur(self):
        self.assertEqual(self.karar({'O': 0.1, 'C': 0.8, 'P': 0.05, 'Q': 0.05}), 'C')

    def test_emin_olmayan_virgul_konmaz(self):
        self.assertEqual(self.karar({'O': 0.35, 'C': 0.55, 'P': 0.05, 'Q': 0.05}), 'O')

    def test_emin_olmayan_soru_noktaya_doner(self):
        self.assertEqual(self.karar({'O': 0.1, 'C': 0.0, 'P': 0.4, 'Q': 0.5}), 'P')


if __name__ == '__main__':
    unittest.main()
