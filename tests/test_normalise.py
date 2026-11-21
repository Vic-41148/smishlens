"""Normaliser: folding rules, masking and disguise flags."""

from smishlens.normalise import deleet_mixed, normalise, normalise_with_flags


def test_confusables_fold_only_in_mixed_tokens():
    assert normalise("pаy now") == "pay now"
    assert normalise("привет") == "привет"


def test_global_mode_folds_everywhere():
    assert "p" in normalise("привет", mode="global")


def test_zero_width_removed():
    assert normalise("fr\u200bee") == "free"


def test_fullwidth_folded():
    assert normalise("ＦＲＥＥ") == "free"


def test_url_masked():
    assert normalise("go to https://x.test/a now") == "go to <url> now"


def test_defanged_url_masked():
    assert normalise("go hxxp://x[.]test/a now") == "go <url> now"


def test_other_tokens_masked():
    assert "<email>" in normalise("mail a@b.com now")
    assert "<money>" in normalise("pay £5 now")
    assert "<phone>" in normalise("call +44 1234 567890")
    assert "<shortcode>" in normalise("text 12345 now")


def test_number_masked():
    assert normalise("meet at 6") == "meet at <num>"


def test_flags_counted():
    text, flags = normalise_with_flags("Ｆree\u200b")
    assert text == "free"
    assert flags == {"fullwidth": 1, "format_chars": 1}


def test_deleet_mixed():
    assert deleet_mixed("fr33") == "free"
