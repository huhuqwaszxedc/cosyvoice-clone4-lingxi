import unittest

from jttts.tts_model.frontend.text_preprocessing import (
    apply_bracket_content_filter,
    normalize_html_entities,
    remove_bracketed_content,
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

    def test_verbalizes_fullwidth_less_than_inside_brackets(self):
        text = "🧠 **正常情况下**：淋巴结很小（直径＜1cm）"

        self.assertEqual(
            verbalize_math_symbols(text),
            "🧠 **正常情况下**：淋巴结很小（直径小于1cm）",
        )

    def test_verbalizes_other_fullwidth_comparison_symbols(self):
        self.assertEqual(
            verbalize_math_symbols("＜1、＞2、＜＝3、＞＝4、≦5、≧6、＝7"),
            "小于1、大于2、小于等于3、大于等于4、小于等于5、大于等于6、等于7",
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

    def test_bracket_filter_is_disabled_by_default(self):
        text = "请播报（这段也要播报）正文"

        self.assertEqual(apply_bracket_content_filter(text), text)

    def test_removes_brackets_and_enclosed_content_when_enabled(self):
        text = "请播报（这段不要播报）正文"

        self.assertEqual(
            apply_bracket_content_filter(text, enabled=True),
            "请播报正文",
        )

    def test_removes_supported_bracket_types(self):
        text = "甲(a)乙[b]丙{c}丁（d）戊【e】己｛f｝庚"

        self.assertEqual(remove_bracketed_content(text), "甲乙丙丁戊己庚")

    def test_removes_nested_and_multiple_bracket_groups(self):
        text = "开始（外层【内层】结束）中间(test)完成"

        self.assertEqual(remove_bracketed_content(text), "开始中间完成")

    def test_preserves_unmatched_brackets(self):
        text = "开始（未结束，正文继续"

        self.assertEqual(remove_bracketed_content(text), text)


if __name__ == "__main__":
    unittest.main()
