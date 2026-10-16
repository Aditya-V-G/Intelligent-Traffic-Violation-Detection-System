from types import SimpleNamespace

from vision.models import find_classes, vehicle_type_map


def m(names):
    return SimpleNamespace(names=dict(enumerate(names)))


def test_negative_class_names():
    for names, bad, good in [
        (["helmet", "no_helmet"], [1], [0]),
        (["With Helmet", "Without Helmet"], [1], [0]),
        (["nohelmet", "helmet", "rider"], [0], [1]),
        (["person-noseatbelt", "person-seatbelt"], [0], [1]),
        (["seatbelt", "no-seatbelt", "windshield"], [1], [0]),
    ]:
        key = "belt" if "belt" in "".join(names) else "helmet"
        assert find_classes(m(names), [key], negative=True) == bad, names
        assert find_classes(m(names), [key], negative=False) == good, names


def test_vehicle_maps():
    coco = m(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"])
    assert vehicle_type_map(coco) == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert vehicle_type_map(m(["mobil", "motor"])) == {0: "car", 1: "motorcycle"}
