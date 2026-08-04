"""The storefront's product tiles are pure functions of article metadata, so
the mapping from a garment to its silhouette, ink and price is testable without
a browser. These caught two real bugs: "Leggings/Tights" drawing as a sock
(because "tights" matched before "legging"), and dark garments rendering an
invisible dark silhouette on their own colour field.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import storefront_ui as shop  # noqa: E402


@pytest.mark.parametrize(
    "product_type,expected",
    [
        ("Trousers", "trousers"),
        ("Leggings/Tights", "trousers"),  # not "sock" — both words are in the name
        ("Socks", "sock"),
        ("Dress", "dress"),
        ("Jumpsuit/Playsuit", "dress"),
        ("Sweater", "top"),
        ("T-shirt", "top"),
        ("Vest top", "top"),
        ("Hoodie", "hoodie"),
        ("Blazer", "outerwear"),
        ("Cardigan", "outerwear"),
        ("Shorts", "shorts"),
        ("Skirt", "skirt"),
        ("Sneakers", "shoe"),
        ("Bra", "bra"),
        ("Bikini top", "bra"),  # before the generic "top"
        ("Swimwear bottom", "underwear"),
        ("Bag", "bag"),
        ("Earring", "accessory"),
    ],
)
def test_product_type_picks_its_silhouette(product_type, expected):
    assert shop.garment_key({"product_type_name": product_type}) == expected


def test_falls_back_to_product_group_when_type_is_unrecognised():
    item = {"product_type_name": "Nonsense Garment", "product_group_name": "Garment Full body"}
    assert shop.garment_key(item) == "dress"


def test_falls_back_to_hanger_when_nothing_matches():
    assert shop.garment_key({}) == "hanger"


def test_every_silhouette_key_has_a_drawing():
    """A mapping entry pointing at a missing shape would raise a KeyError deep
    inside the tile renderer, where it is hard to trace back."""
    referenced = {shape for _, shape in shop._TYPE_KEYWORDS} | set(shop._GROUP_MAP.values())
    assert referenced <= set(shop._GARMENTS)


def test_group_map_covers_the_datasets_whole_vocabulary():
    dataset_groups = {
        "Garment Upper body", "Garment Lower body", "Garment Full body",
        "Accessories", "Underwear", "Shoes", "Swimwear", "Socks & Tights",
        "Nightwear", "Unknown", "Cosmetic", "Bags",
    }
    assert dataset_groups <= set(shop._GROUP_MAP)


@pytest.mark.parametrize(
    "hex_colour,expected",
    [("#1F1F23", "#FFFFFF"), ("#F4F4F5", shop.HM["ink"]), ("#3B82F6", "#FFFFFF")],
)
def test_ink_contrasts_with_the_colour_field(hex_colour, expected):
    assert shop.ink_for(hex_colour) == expected


def test_price_formatting():
    assert shop.fmt_price(19.99) == "19.99"
    assert shop.fmt_price(1234.5) == "1,234.50"
    assert shop.fmt_price(None) == "—"
    assert shop.fmt_price("not a number") == "—"
