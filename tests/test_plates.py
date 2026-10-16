from vision.plates import PlateVoter, _join_lines, correct_plate, format_plate, is_valid_plate


def test_valid_formats():
    for p in ["KA09AB1234", "MH12A1234", "DL3CAB1234", "TN01ABC1234", "22BH1234AA", "UP16Z0001"]:
        assert is_valid_plate(p), p
    for p in ["XX09AB1234", "KA09AB123", "KA09AB12345", "1234", "", "KAO9AB1234"]:
        assert not is_valid_plate(p), p


def test_position_aware_correction():
    assert correct_plate("KA O9 AB 1234")[0] == "KA09AB1234"      # O in a digit slot
    assert correct_plate("KA09A81234")[0] == "KA09AB1234"         # 8 in a letter slot
    assert correct_plate("MH12AB I234")[0] == "MH12AB1234"
    assert correct_plate("IND KA09AB1234")[0] == "KA09AB1234"     # hologram text
    assert correct_plate("kao9-ab.1234")[0] == "KA09AB1234"


def test_rejects_garbage_instead_of_guessing():
    for raw in ["HELLO", "1234567", "ABCDEFGHIJ", "QQ99ZZ9999", "", "KA0OOOOO1"]:
        assert correct_plate(raw)[0] is None, raw


def test_too_many_fixes_rejected():
    assert correct_plate("KAOOABIZS4")[0] is None    # 4+ substitutions needed


def test_two_line_plate_joined_top_to_bottom():
    res = [  # EasyOCR (box, text, conf); bottom line returned first
        ([[0, 40], [100, 40], [100, 70], [0, 70]], "AB1234", 0.9),
        ([[10, 0], [90, 0], [90, 30], [10, 30]], "KA09", 0.8),
    ]
    text, conf = _join_lines(res)
    assert text == "KA09AB1234"
    assert abs(conf - 0.85) < 1e-6


def test_voting_agreement():
    v = PlateVoter()
    for p, w in [("KA09AB1234", 0.7), ("KA09AB1234", 0.6), ("KA09AB1284", 0.3)]:
        v.add(p, w)
    assert v.result()[0] == "KA09AB1234"


def test_voting_char_level():
    v = PlateVoter()   # never read identically twice, but each position agrees
    for p in ["KA09AB1234", "KA09AB1234", "KA08AB1234", "KA09AB1294"]:
        v.add(p, 0.5)
    assert v.result()[0] == "KA09AB1234"


def test_voting_disagreement_is_not_readable():
    v = PlateVoter()
    v.add("KA09AB1234", 0.5)
    v.add("MH12CD5678", 0.5)
    assert v.result()[0] is None
    single = PlateVoter()
    single.add("KA09AB1234", 0.4)          # one weak read is not enough
    assert single.result()[0] is None
    assert PlateVoter().result()[0] is None


def test_format():
    assert format_plate("KA09AB1234") == "KA 09 AB 1234"
    assert format_plate("22BH1234AA") == "22 BH 1234 AA"
