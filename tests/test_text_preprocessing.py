import unittest

from jttts.tts_model.frontend.text_preprocessing import (
    normalize_html_entities,
    strip_bracket_marks,
    verbalize_math_symbols,
)


class TextPreprocessingTest(unittest.TestCase):
    def test_removes_complete_quote_entities(self):
        text = "请回复序号选择要拨打的号码（如&ldquo;选1&rdquo;）～"

        self.assertEqual(
            normalize_html_entities(text),
            "请回复序号选择要拨打的号码（如选1）～",
        )

    def test_removes_quote_entity_without_semicolon(self):
        text = "请回复序号选择要拨打的号码如&ldquo;选1&rdquo～"

        self.assertEqual(
            normalize_html_entities(text),
            "请回复序号选择要拨打的号码如选1～",
        )

    def test_does_not_repair_unknown_ampersand_text(self):
        self.assertEqual(normalize_html_entities("A&B～"), "A&B～")

    def test_decodes_and_verbalizes_less_than_or_equal(self):
        text = normalize_html_entities("- 南风&le;3级，注意防暑防晒～")

        self.assertEqual(
            verbalize_math_symbols(text),
            "- 南风小于等于3级，注意防暑防晒～",
        )

    def test_verbalizes_comparison_entities(self):
        text = "&lt;1、&gt;2、&le;3、&ge;4、&ne;5"

        self.assertEqual(
            verbalize_math_symbols(normalize_html_entities(text)),
            "小于1、大于2、小于等于3、大于等于4、不等于5",
        )

    def test_verbalizes_arithmetic_entities(self):
        text = "2&times;3、6&divide;2、&plusmn;5"

        self.assertEqual(
            verbalize_math_symbols(normalize_html_entities(text)),
            "2乘3、6除2、正负5",
        )

    def test_keeps_layout_and_unit_entities_unchanged(self):
        text = "&nbsp;&hellip;&mdash;&ndash;&deg;"

        self.assertEqual(normalize_html_entities(text), text)

    def test_keeps_content_inside_brackets(self):
        self.assertEqual(strip_bracket_marks("号码（如“选1”）"), "号码如“选1”")


if __name__ == "__main__":
    unittest.main()
