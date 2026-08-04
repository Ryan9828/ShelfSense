"""A fast-fashion product-listing page, rendered from recommender output.

Deliberately a *second*, separate design system from `ui.py`. That one is the
analyst view — cards, stat tiles, blue accent, rounded corners. This one
pastiches a high-street clothing retailer's PLP: square corners, near-black
ink on white, one red accent, uppercase letter-spaced labels, a dense
portrait grid. Keeping them in separate modules stops the two vocabularies
leaking into each other.

**Why it looks like this.** H&M's own site blocks automated access, so the
tokens here come from published brand references rather than a measurement of
the live site: brand red #CC071E (PMS 1795 C), white ground, and the
Helvetica-family sans their site falls back to behind the proprietary H&M
Sans. Layout follows fast-fashion PLP convention — 3:4 portrait tiles, tight
gutters, product name over price, zero border radius.

**Why it is not branded as H&M.** This is a ShelfSense demo that happens to
run on H&M's public dataset, so it uses ShelfSense's own wordmark in that
visual idiom rather than reproducing another company's logo. The nav links are
inert scenery — the page is a shop's *front* page for one customer, so "Home"
is the only one that describes what is actually on screen.

**Why there are no photographs.** The Kaggle competition does ship product
images (~25GB), but this project never downloads them — see
`scripts/download_data.sh`. Each tile therefore renders the garment's real
`colour_group_name` as a full-bleed colour field with a line-art silhouette
chosen from its `product_group_name` / `product_type_name`. Everything on a
tile is real data; nothing is a stand-in image.
"""
import html

import streamlit as st

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
HM = {
    "red": "#CC071E",       # brand red, PMS 1795 C
    "ink": "#222222",       # body copy — the site is near-black, not pure black
    "ink_soft": "#767676",  # secondary product metadata
    "rule": "#E8E8E8",
    "ground": "#FFFFFF",
}

# The shop nav. These 12 items are the shop's front page for one customer, not
# a department listing, so "Home" is the honest label — an underlined
# "Ladieswear" implied the visitor had browsed into a category, which they
# hadn't. The rest are inert scenery, present so the bar reads as a shop.
NAV = ["Home", "New in", "Clothing", "Sale"]

# Line-art garment silhouettes, drawn on a 48x64 (3:4) canvas to match the
# portrait tile. Stroked rather than filled so they read as fashion line
# drawings and inherit the contrast ink chosen per colour.
_GARMENTS = {
    "top": '<path d="M17 13 L9 17 L5 27 L11 30 L14 25 L14 52 L34 52 L34 25 L37 30 L43 27 L39 17 L31 13 C29 16 27 17 24 17 C21 17 19 16 17 13 Z"/>',
    "outerwear": '<path d="M17 13 L9 17 L5 27 L11 30 L14 25 L14 52 L34 52 L34 25 L37 30 L43 27 L39 17 L31 13 C29 16 27 17 24 17 C21 17 19 16 17 13 Z"/><path d="M24 17 V52"/>',
    "hoodie": '<path d="M17 13 L9 17 L5 27 L11 30 L14 25 L14 52 L34 52 L34 25 L37 30 L43 27 L39 17 L31 13 Z"/><path d="M17 13 C19 20 29 20 31 13"/><path d="M24 20 V52"/>',
    "trousers": '<path d="M15 13 H33 L35 53 H27 L24 32 L21 53 H13 Z"/><path d="M15 19 H33"/>',
    "shorts": '<path d="M15 15 H33 L34 41 H27 L24 28 L21 41 H14 Z"/><path d="M15 21 H33"/>',
    "skirt": '<path d="M16 14 H32 L38 51 H10 Z"/><path d="M16 20 H32"/>',
    "dress": '<path d="M17 13 L10 18 L13 23 L16 20 L11 52 H37 L32 20 L35 23 L38 18 L31 13 C29 16 27 17 24 17 C21 17 19 16 17 13 Z"/>',
    "shoe": '<path d="M9 42 V33 C9 31 11 30 13 30 H17 L23 36 C25 38 29 39 34 40 L40 41 C42 42 43 43 43 45 V47 H9 Z"/>',
    "bag": '<path d="M12 25 H36 L38 52 H10 Z"/><path d="M18 25 V20 C18 15 21 12 24 12 C27 12 30 15 30 20 V25"/>',
    "underwear": '<path d="M13 24 H35 L33 34 C29 35 26 39 25 45 H23 C22 39 19 35 15 34 Z"/>',
    "bra": '<path d="M9 25 C9 21 13 19 17 19 C21 19 24 22 24 26 C24 22 27 19 31 19 C35 19 39 21 39 25 C39 33 33 38 24 38 C15 38 9 33 9 25 Z"/>',
    "sock": '<path d="M19 12 H29 V33 C29 39 35 41 35 46 C35 50 31 53 27 53 C21 53 17 49 17 44 V12 Z"/>',
    "accessory": '<path d="M24 14 C25.5 14 27 15.5 27 17 C27 18.5 25.5 20 24 20 C22.5 20 21 18.5 21 17 C21 15.5 22.5 14 24 14 Z"/><path d="M24 20 C29 25 33 31 33 37 C33 43 29 47 24 47 C19 47 15 43 15 37 C15 31 19 25 24 20 Z"/>',
    "cosmetic": '<path d="M20 13 H28 V19 L32 25 V51 H16 V25 L20 19 Z"/><path d="M16 32 H32"/>',
    "hanger": '<path d="M24 15 C21.5 15 20 16.5 20 18.5 C20 20.5 22 21.5 24 22.5 V26"/><path d="M7 41 L24 26 L41 41 C43 42.5 42 45 40 45 H8 C6 45 5 42.5 7 41 Z"/>',
}

# Checked in order against product_type_name — first hit wins, so the more
# specific entries have to come before the generic ones ("bikini top" before
# "top", "swimwear bottom" before "bottom").
_TYPE_KEYWORDS = [
    ("bikini top", "bra"), ("swimwear top", "bra"), ("bra", "bra"),
    ("swimwear bottom", "underwear"), ("bikini bottom", "underwear"),
    ("underwear bottom", "underwear"), ("brief", "underwear"), ("thong", "underwear"),
    # "legging" before "tights": the type name is "Leggings/Tights", which
    # contains both, and it should draw as legwear rather than as a sock.
    ("legging", "trousers"), ("sock", "sock"), ("tights", "sock"),
    ("hoodie", "hoodie"), ("sweatshirt", "hoodie"),
    ("jacket", "outerwear"), ("blazer", "outerwear"), ("coat", "outerwear"),
    ("cardigan", "outerwear"), ("gilet", "outerwear"),
    ("shorts", "shorts"), ("skirt", "skirt"),
    ("trouser", "trousers"), ("jeans", "trousers"), ("chino", "trousers"),
    ("dress", "dress"), ("jumpsuit", "dress"), ("playsuit", "dress"),
    ("sneaker", "shoe"), ("shoe", "shoe"), ("boot", "shoe"), ("sandal", "shoe"),
    ("slipper", "shoe"), ("heel", "shoe"),
    ("bag", "bag"), ("backpack", "bag"), ("purse", "bag"),
    ("earring", "accessory"), ("necklace", "accessory"), ("ring", "accessory"),
    ("bracelet", "accessory"), ("hat", "accessory"), ("cap", "accessory"),
    ("scarf", "accessory"), ("belt", "accessory"), ("glove", "accessory"),
    ("sunglasses", "accessory"), ("hair", "accessory"),
    ("shirt", "top"), ("t-shirt", "top"), ("blouse", "top"), ("sweater", "top"),
    ("vest top", "top"), ("top", "top"), ("pyjama", "top"),
]

# product_group_name is a closed 12-value vocabulary, so this is the exhaustive
# fallback when no product_type keyword matched.
_GROUP_MAP = {
    "Garment Upper body": "top",
    "Garment Lower body": "trousers",
    "Garment Full body": "dress",
    "Accessories": "accessory",
    "Underwear": "underwear",
    "Shoes": "shoe",
    "Swimwear": "bra",
    "Socks & Tights": "sock",
    "Nightwear": "top",
    "Bags": "bag",
    "Cosmetic": "cosmetic",
    "Unknown": "hanger",
}


def garment_key(item: dict) -> str:
    ptype = str(item.get("product_type_name") or "").lower()
    for keyword, shape in _TYPE_KEYWORDS:
        if keyword in ptype:
            return shape
    return _GROUP_MAP.get(str(item.get("product_group_name") or ""), "hanger")


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance, used only to pick a readable ink over the
    garment's colour field — a black silhouette on Black is invisible."""
    h = hex_colour.lstrip("#")
    srgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink_for(hex_colour: str) -> str:
    return "#FFFFFF" if _relative_luminance(hex_colour) < 0.45 else HM["ink"]


def fmt_price(price) -> str:
    """The dataset's own price, rescaled at build time (see
    data.attach_retail_price). No currency symbol: the competition data does not
    say which currency these are, and inventing one would be the only made-up
    thing on the tile."""
    if price is None:
        return "—"
    try:
        return f"{float(price):,.2f}"
    except (TypeError, ValueError):
        return "—"


def store_css() -> None:
    """Inject the storefront's own stylesheet. Scoped under .hm-* so it cannot
    collide with the analyst design system in ui.py."""
    st.markdown(
        f"""
<style>
.hm-store {{
    background: {HM["ground"]};
    border: 1px solid {HM["rule"]};
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: {HM["ink"]};
    margin: 4px 0 0 0;
}}

/* --- shop chrome ---------------------------------------------------- */
.hm-chrome {{
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    padding: 14px 18px; border-bottom: 1px solid {HM["rule"]};
}}
.hm-brand {{
    font-weight: 700; font-size: 1.35rem; letter-spacing: -0.02em;
    color: {HM["red"]}; font-style: italic; flex: none;
}}
.hm-nav {{
    display: flex; gap: 16px; flex-wrap: wrap; margin-left: 4px;
    font-size: 0.82rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: {HM["ink"]};
}}
.hm-nav span {{ white-space: nowrap; padding-bottom: 2px; }}
.hm-nav .on {{ border-bottom: 2px solid {HM["red"]}; font-weight: 700; }}
.hm-chrome .hm-spacer {{ flex: 1 1 auto; }}
.hm-chrome .hm-glyphs {{ display: flex; gap: 12px; color: {HM["ink_soft"]}; flex: none; }}

/* --- editorial banner ----------------------------------------------- */
.hm-banner {{
    padding: 22px 18px 4px 18px;
}}
.hm-banner .hm-eyebrow {{
    font-size: 0.76rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: {HM["ink_soft"]};
}}
.hm-banner h3 {{
    font-size: 1.62rem; font-weight: 700; letter-spacing: -0.01em;
    margin: 4px 0 0 0; padding: 0; color: {HM["ink"]};
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}}
.hm-banner .hm-sub {{ font-size: 0.94rem; color: {HM["ink_soft"]}; margin-top: 4px; }}

/* --- product grid --------------------------------------------------- */
/* Explicit column counts rather than auto-fill: there are always 12 tiles, and
   2/3/4 all divide it evenly, so no row is ever left with orphans. */
.hm-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 26px 12px; padding: 18px;
}}
@media (min-width: 640px) {{ .hm-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (min-width: 1000px) {{ .hm-grid {{ grid-template-columns: repeat(4, 1fr); }} }}
.hm-tile {{ display: flex; flex-direction: column; }}
.hm-shot {{
    position: relative; aspect-ratio: 3 / 4; width: 100%;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}}
.hm-shot svg {{ width: 62%; height: 62%; opacity: 0.62; }}
.hm-rank {{
    position: absolute; top: 7px; left: 7px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 2px 5px; background: rgba(255,255,255,0.86); color: {HM["ink"]};
    font-variant-numeric: tabular-nums;
}}
.hm-flag {{
    position: absolute; bottom: 0; left: 0; right: 0;
    background: {HM["red"]}; color: #fff; text-align: center;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 3px 4px;
}}
.hm-meta {{ padding: 8px 2px 0 2px; }}
.hm-name {{
    font-size: 0.96rem; font-weight: 400; color: {HM["ink"]}; line-height: 1.35;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.hm-type {{
    font-size: 0.86rem; color: {HM["ink_soft"]}; margin-top: 2px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.hm-price {{ font-size: 0.96rem; font-weight: 700; margin-top: 5px; color: {HM["ink"]}; }}

/* --- model tabs styled as shop category tabs ------------------------ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0; border-bottom: 1px solid {HM["rule"]};
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.85rem; letter-spacing: 0.11em; text-transform: uppercase;
    font-weight: 600; padding: 12px 20px; color: {HM["ink_soft"]};
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color: {HM["ink"]}; }}
/* pointer-events: none is load-bearing — the highlight is an absolutely
   positioned bar sitting over the active tab, and without this it swallows
   clicks aimed at that tab. */
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background: {HM["red"]}; pointer-events: none;
}}

</style>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def _tile(item: dict, rank: int, swatch: str, flag: str | None) -> str:
    colour = str(item.get("colour_group_name") or "")
    bg = swatch
    silhouette = _GARMENTS[garment_key(item)]
    name = html.escape(str(item.get("prod_name") or item.get("article_id") or "?"))
    ptype = html.escape(str(item.get("product_type_name") or ""))
    flag_html = f'<div class="hm-flag">{html.escape(flag)}</div>' if flag else ""
    return (
        f'<article class="hm-tile">'
        f'<div class="hm-shot" style="background:{bg}" title="{html.escape(colour)}">'
        f'<svg viewBox="0 0 48 64" fill="none" stroke="{ink_for(bg)}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{silhouette}</svg>'
        f'<span class="hm-rank">{rank:02d}</span>{flag_html}</div>'
        f'<div class="hm-meta"><div class="hm-name">{name}</div>'
        f'<div class="hm-type">{ptype}</div>'
        f'<div class="hm-price">{fmt_price(item.get("price"))}</div></div>'
        f"</article>"
    )


def store(
    items: list[dict],
    swatch_of,
    eyebrow: str,
    headline: str,
    subline: str,
    flag_ids: set[str] | None = None,
    flag_text: str = "Picked for you",
) -> None:
    """Render the whole shop panel in one markdown call.

    `swatch_of` maps a colour_group_name to a hex code (reuses ui.swatch_hex so
    the two views agree on what "Dark Blue" looks like). `flag_ids` are article
    ids to tag — the storefront's way of showing which items personalization
    added versus which the popularity baseline would have shown anyway.
    """
    flag_ids = flag_ids or set()
    nav = "".join(
        f'<span class="{"on" if i == 0 else ""}">{html.escape(d)}</span>'
        for i, d in enumerate(NAV)
    )

    tiles = "".join(
        _tile(
            item,
            i + 1,
            swatch_of(str(item.get("colour_group_name") or "")),
            flag_text if str(item.get("article_id")) in flag_ids else None,
        )
        for i, item in enumerate(items)
    )

    st.markdown(
        f'<div class="hm-store">'
        f'<div class="hm-chrome"><div class="hm-brand">ShelfSense</div>'
        f'<nav class="hm-nav">{nav}</nav><div class="hm-spacer"></div>'
        f'<div class="hm-glyphs">♡ ⌕ ⌂</div></div>'
        f'<div class="hm-banner"><div class="hm-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h3>{html.escape(headline)}</h3>"
        f'<div class="hm-sub">{html.escape(subline)}</div></div>'
        f'<div class="hm-grid">{tiles}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
