"""tests/test_i18n.py — Unit tests for the i18n translation layer."""

import unittest

from chaotic_pfc._i18n import t


class TestI18n(unittest.TestCase):
    def test_pt_returns_portuguese(self):
        title = t("attractor.standard", lang="pt")
        self.assertIn("Hénon", title)
        self.assertIn("Padrão", title)

    def test_en_returns_english(self):
        title = t("attractor.standard", lang="en")
        self.assertIn("Hénon", title)
        self.assertIn("Standard", title)

    def test_unknown_key_returns_key(self):
        self.assertEqual(t("nonexistent.key", lang="pt"), "nonexistent.key")

    def test_unknown_lang_returns_key(self):
        self.assertEqual(t("attractor.standard", lang="fr"), "attractor.standard")

    def test_comm_keys_bilingual(self):
        for key in ("comm.ideal", "comm.fir", "comm.order_n"):
            en = t(key, lang="en")
            pt = t(key, lang="pt")
            self.assertNotEqual(en, pt)
            self.assertTrue(len(en) > 0 and len(pt) > 0)
