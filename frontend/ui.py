"""Shared design system for the ShelfSense frontend.

Single source of truth for colors, fonts, and the HTML/CSS components every
page uses (hero header, product cards, badges, callouts, stat tiles).
Streamlit-native theme tokens live in .streamlit/config.toml; everything that
config version can't express (custom fonts, cards, hover states) is injected
here by apply_theme().

Design system: monochrome + blue accent, Rubik (headings) / Nunito Sans (body).
Chart palette (teal/blue/amber) is CVD-validated — see views/model_comparison.py.
API calls live in frontend/api.py, not here.
"""
import html

import streamlit as st

REPO_URL = "https://github.com/Ryan9828/ShelfSense"

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#18181B",
    "secondary": "#3F3F46",
    "accent": "#2563EB",
    "accent_soft": "#EFF4FE",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "foreground": "#09090B",
    "muted_text": "#52525B",
    "faint_text": "#71717A",
    "border": "#E4E4E7",
    "teal": "#0D9488",
    "amber": "#D97706",
    "amber_soft": "#FDF3E3",
    "teal_soft": "#E7F6F4",
}

# Lucide icons (https://lucide.dev, ISC license) — inline SVG so no emoji are
# used as structural icons and the strokes inherit currentColor.
_ICON_PATHS = {
    "logo": '<path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "flask": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "sparkles": '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M8 17v-3"/><path d="M13 17V9"/><path d="M18 17V5"/>',
    "pointer": '<path d="m9 9 5 12 1.8-5.2L21 14Z"/><path d="M7.2 2.2 8 5.1"/><path d="m5.1 8-2.9-.8"/><path d="M14 4.1 12 6"/><path d="m6 12-1.9 2"/>',
    "route": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
    "github": '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "bag": '<path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m6.08 10.37-3.5 1.6a1 1 0 0 0 0 1.81l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9a1 1 0 0 0 0-1.83l-3.5-1.59"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 15) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" width="{size}" height="{size}">'
        f"{_ICON_PATHS[name]}</svg>"
    )


ICONS = {name: icon(name) for name in _ICON_PATHS}
ICONS["logo"] = icon("logo", 26)


def apply_theme() -> None:
    """Inject fonts + component CSS. Call once per page run (idempotent)."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700&family=Rubik:wght@500;600;700&display=swap');

html, body, .stApp, [class*="st-"] {{
    font-family: 'Nunito Sans', -apple-system, 'Segoe UI', sans-serif;
}}

/* Light-mode guard. The .ss-* components below are drawn on a light surface,
   so a host that starts the app in dark mode — a system dark-mode preference
   where .streamlit/config.toml never made it into the deploy — would frame
   light cards in dark chrome. config.toml is the real fix; this keeps the
   page coherent if it is ever missing. */
:root, .stApp {{
    --background-color: {COLORS["background"]};
    --secondary-background-color: {COLORS["surface"]};
    --text-color: {COLORS["foreground"]};
    color-scheme: light;
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: {COLORS["background"]}; color: {COLORS["foreground"]};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {COLORS["surface"]}; }}
[data-testid="stSidebarNav"] a span, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, .stMarkdown, .stMarkdown p, .stMarkdown li {{
    color: {COLORS["foreground"]};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {COLORS["muted_text"]} !important;
}}

/* Cap the measure: full-width text on a 27" monitor is unreadable, but the
   storefront's three model columns still need room to sit side by side — so
   the container stays wide and prose is capped separately. :not([class])
   limits this to plain markdown, leaving the .ss-* components their own
   widths; inside a column the column is narrower anyway, so it's a no-op. */
.block-container {{ max-width: 1240px; padding-top: 2.6rem; }}
[data-testid="stMain"] .stMarkdown p:not([class]),
[data-testid="stMain"] .stMarkdown li {{ max-width: 82ch; }}
/* The broad selector above must not clobber Streamlit's icon font (material
   symbols render as ligature text and turn into garbled words without this) */
span[data-testid="stIconMaterial"], [class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Symbols Sharp' !important;
}}
code, pre, kbd {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace !important; }}
h1, h2, h3, h4, .ss-font-heading {{
    font-family: 'Rubik', -apple-system, 'Segoe UI', sans-serif !important;
    letter-spacing: -0.01em;
}}

/* Replace the default rainbow decoration strip with a brand accent line */
[data-testid="stDecoration"] {{
    background: linear-gradient(90deg, {COLORS["primary"]}, {COLORS["accent"]});
    height: 3px;
}}
[data-testid="stSidebar"] {{ border-right: 1px solid {COLORS["border"]}; }}

/* Buttons: radius, weight, motion (150-300ms), visible focus */
.stButton > button, .stDownloadButton > button {{
    border-radius: 8px;
    font-weight: 700;
    font-family: 'Nunito Sans', sans-serif;
    transition: background-color 200ms ease, border-color 200ms ease,
                color 200ms ease, box-shadow 200ms ease;
}}
.stButton > button:hover {{ border-color: {COLORS["accent"]}; color: {COLORS["accent"]}; }}
.stButton > button[kind="primary"]:hover {{ background-color: #1D4FD7; color: #fff; }}
.stButton > button:focus-visible {{ outline: 3px solid {COLORS["accent"]}55; outline-offset: 1px; }}

/* Hero header */
.ss-hero {{ display: flex; align-items: center; gap: 14px; margin: 0 0 4px 0; }}
.ss-hero-mark {{
    width: 46px; height: 46px; flex: none; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    background: {COLORS["primary"]}; color: #fff;
}}
.ss-hero h1 {{
    font-family: 'Rubik', sans-serif; font-size: 1.72rem; font-weight: 700;
    color: {COLORS["foreground"]}; margin: 0; padding: 0; line-height: 1.2;
}}
.ss-hero .ss-kicker {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: {COLORS["accent"]}; margin-bottom: 2px;
}}
.ss-sub {{ color: {COLORS["muted_text"]}; font-size: 0.95rem; line-height: 1.55; max-width: 72ch; margin: 6px 0 20px 0; }}

/* Model column header */
.ss-colhead {{
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    padding: 10px 14px; border: 1px solid {COLORS["border"]}; border-radius: 10px;
    background: {COLORS["surface"]}; margin-bottom: 10px;
}}
.ss-colhead .ss-model {{ font-family: 'Rubik', sans-serif; font-weight: 600; font-size: 1rem; color: {COLORS["foreground"]}; }}

/* Badges */
.ss-badge {{
    display: inline-flex; align-items: center; white-space: nowrap;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 3px 9px; border-radius: 999px;
}}
.ss-badge-shipped {{ background: {COLORS["accent_soft"]}; color: #1D4FD7; }}
.ss-badge-control {{ background: {COLORS["teal_soft"]}; color: #0B7A70; }}
.ss-badge-bench   {{ background: {COLORS["amber_soft"]}; color: #B45309; }}

/* Product cards */
.ss-card {{
    display: flex; align-items: center; gap: 11px;
    background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]};
    border-radius: 10px; padding: 10px 12px; margin-bottom: 8px;
    transition: border-color 200ms ease, box-shadow 200ms ease;
}}
.ss-card:hover {{ border-color: #A1A1AA; box-shadow: 0 2px 10px rgba(9, 9, 11, 0.06); }}
.ss-card .ss-rank {{
    flex: none; width: 20px; font-size: 0.72rem; font-weight: 700;
    color: {COLORS["faint_text"]}; font-variant-numeric: tabular-nums; text-align: right;
}}
.ss-swatch {{
    flex: none; width: 14px; height: 14px; border-radius: 5px;
    border: 1px solid rgba(9, 9, 11, 0.14);
}}
.ss-card .ss-body {{ min-width: 0; }}
.ss-card .ss-name {{
    font-weight: 700; font-size: 0.9rem; color: {COLORS["foreground"]};
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.ss-card .ss-meta {{ font-size: 0.78rem; color: {COLORS["muted_text"]}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

/* Callouts (replaces st.info so iconography + type stay on-system) */
.ss-callout {{
    display: flex; gap: 10px; align-items: flex-start;
    border: 1px solid {COLORS["border"]}; border-left: 3px solid {COLORS["accent"]};
    background: {COLORS["surface"]}; border-radius: 8px;
    padding: 12px 14px; margin: 8px 0; font-size: 0.88rem; line-height: 1.55;
    color: {COLORS["secondary"]};
}}
.ss-callout svg {{ flex: none; margin-top: 2px; color: {COLORS["accent"]}; }}
.ss-callout-amber {{ border-left-color: {COLORS["amber"]}; }}
.ss-callout-amber svg {{ color: {COLORS["amber"]}; }}

/* Stat tile */
.ss-stat {{
    display: inline-block; background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]}; border-radius: 12px;
    padding: 14px 22px; margin: 4px 0 8px 0;
}}
.ss-stat .ss-stat-label {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {COLORS["faint_text"]};
}}
.ss-stat .ss-stat-value {{
    font-family: 'Rubik', sans-serif; font-size: 1.9rem; font-weight: 700;
    color: {COLORS["foreground"]}; font-variant-numeric: tabular-nums; line-height: 1.25;
}}

/* Home: feature cards for the page directory */
.ss-feature {{
    background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]};
    border-radius: 12px; padding: 18px; margin-bottom: 6px;
    transition: border-color 200ms ease, box-shadow 200ms ease;
}}
.ss-feature:hover {{ border-color: #A1A1AA; box-shadow: 0 2px 10px rgba(9, 9, 11, 0.06); }}
/* Side-by-side on desktop the three cards must be equal height, or the CTA
   links below them sit at three different heights.

   Streamlit already stretches the columns themselves to the tallest one, but
   the block wrapper inside each column is display:block with height:auto, so
   that stretch stops dead before it reaches the card. Thread a definite
   height down the whole chain — wrapper, the card's own element container,
   the markdown wrappers — and let the card fill what's left after the page
   link. min-height is the fallback for engines without :has(). */
@media (min-width: 768px) {{
    .ss-feature {{ min-height: 210px; }}
    [data-testid="stColumn"]:has(.ss-feature) > [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stColumn"]:has(.ss-feature) > [data-testid="stVerticalBlockBorderWrapper"] > div,
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stVerticalBlock"] {{
        height: 100%;
    }}
    /* Percentage heights can't do the rest of the job — each wrapper would be
       sized from the content it is itself supposed to be stretching — so the
       remaining levels are a plain flex chain: every wrapper grows into the
       spare space (flex-grow 1) and none shrinks below its own copy
       (flex-shrink 0, or the tallest card spills through its own border). */
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stElementContainer"]:has(.ss-feature),
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stMarkdown"]:has(.ss-feature),
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stMarkdownContainer"]:has(.ss-feature),
    [data-testid="stColumn"]:has(.ss-feature) .ss-feature {{
        flex: 1 0 auto;
    }}
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stElementContainer"]:has(.ss-feature),
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stMarkdown"]:has(.ss-feature),
    [data-testid="stColumn"]:has(.ss-feature) [data-testid="stMarkdownContainer"]:has(.ss-feature) {{
        display: flex; flex-direction: column;
    }}
    [data-testid="stColumn"]:has(.ss-feature) .ss-feature {{
        margin-bottom: 0; box-sizing: border-box;
    }}
}}
.ss-feat-icon {{
    width: 38px; height: 38px; border-radius: 10px;
    background: {COLORS["accent_soft"]}; color: {COLORS["accent"]};
    display: flex; align-items: center; justify-content: center; margin-bottom: 12px;
}}
.ss-feat-title {{
    font-family: 'Rubik', sans-serif; font-weight: 600; font-size: 1.02rem;
    color: {COLORS["foreground"]}; margin: 0 0 6px 0;
}}
.ss-feat-desc {{ font-size: 0.87rem; color: {COLORS["muted_text"]}; line-height: 1.55; margin: 0; }}

/* Home: numbered routing steps */
.ss-step {{
    display: flex; gap: 12px; align-items: flex-start;
    background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]};
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; font-size: 0.9rem;
    line-height: 1.5; color: {COLORS["secondary"]};
}}
.ss-step-num {{
    flex: none; width: 26px; height: 26px; border-radius: 8px;
    background: #F4F4F5; color: {COLORS["secondary"]}; font-weight: 700;
    font-size: 0.8rem; display: flex; align-items: center; justify-content: center;
    font-variant-numeric: tabular-nums;
}}
.ss-step strong {{ color: {COLORS["foreground"]}; }}

/* st.page_link CTAs on the home cards (sidebar nav uses its own testid) */
[data-testid="stPageLink-NavLink"] {{ border-radius: 8px; }}
[data-testid="stPageLink-NavLink"] p {{ color: {COLORS["accent"]}; font-weight: 700; }}
[data-testid="stPageLink-NavLink"]:hover p {{ text-decoration: underline; }}

/* Verdict rows on the comparison page */
.ss-verdict {{
    display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
    background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]};
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; font-size: 0.9rem;
}}
.ss-verdict .ss-ci {{ color: {COLORS["muted_text"]}; font-variant-numeric: tabular-nums; }}

/* Stat card grid — reflows to one column on mobile instead of squashing */
.ss-cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px; margin: 10px 0 6px 0;
}}
.ss-cards .ss-stat {{ display: block; margin: 0; padding: 14px 16px; }}
.ss-stat .ss-stat-sub {{
    font-size: 0.79rem; color: {COLORS["muted_text"]}; line-height: 1.45; margin-top: 2px;
}}
.ss-stat-accent .ss-stat-value {{ color: {COLORS["accent"]}; }}

/* Spec pills — the at-a-glance "what this project is" row */
.ss-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px 0; }}
.ss-pill {{
    background: #F4F4F5; border: 1px solid {COLORS["border"]}; color: {COLORS["secondary"]};
    border-radius: 999px; padding: 4px 12px; font-size: 0.79rem; font-weight: 600;
}}

/* Identifier chip — the raw 64-char customer hash, shown deliberately small
   and monospaced so it reads as provenance rather than as a headline */
.ss-idchip {{
    display: inline-flex; align-items: center; gap: 7px;
    font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 0.74rem;
    color: {COLORS["faint_text"]}; background: #F4F4F5;
    border: 1px solid {COLORS["border"]}; border-radius: 7px; padding: 3px 9px;
}}
.ss-idchip .ss-idchip-key {{ font-family: 'Nunito Sans', sans-serif; font-weight: 700; color: {COLORS["faint_text"]}; }}

/* Section heading with a rule, so long explanatory pages stay scannable */
.ss-section {{
    display: flex; align-items: center; gap: 9px;
    margin: 26px 0 2px 0; padding-bottom: 8px;
    border-bottom: 1px solid {COLORS["border"]};
}}
.ss-section svg {{ color: {COLORS["accent"]}; flex: none; }}
.ss-section h2 {{
    font-family: 'Rubik', sans-serif; font-size: 1.12rem; font-weight: 600;
    color: {COLORS["foreground"]}; margin: 0; padding: 0;
}}

/* Compact card variant for dense strips (purchase history, 4-up rec grids) */
.ss-card-compact {{ padding: 8px 10px; gap: 9px; margin-bottom: 6px; }}
.ss-card-compact .ss-name {{ font-size: 0.84rem; }}
.ss-card-compact .ss-meta {{ font-size: 0.73rem; }}

/* Column footnote under each model's list (e.g. overlap with the baseline) */
.ss-colnote {{
    font-size: 0.77rem; color: {COLORS["muted_text"]}; line-height: 1.45;
    padding: 8px 2px 2px 2px;
}}
.ss-colnote strong {{ color: {COLORS["foreground"]}; font-variant-numeric: tabular-nums; }}

.ss-footer {{
    color: {COLORS["faint_text"]}; font-size: 0.83rem; line-height: 1.6;
    border-top: 1px solid {COLORS["border"]}; padding-top: 14px; margin-top: 30px;
}}
.ss-footer a {{ color: {COLORS["accent"]}; }}

@media (prefers-reduced-motion: reduce) {{
    .stApp *, .stApp *::before, .stApp *::after {{
        transition-duration: 0.01ms !important; animation-duration: 0.01ms !important;
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str, kicker: str = "ShelfSense") -> None:
    st.markdown(
        f"""
<div class="ss-hero">
  <div class="ss-hero-mark">{ICONS["logo"]}</div>
  <div>
    <div class="ss-kicker">{html.escape(kicker)}</div>
    <h1>{html.escape(title)}</h1>
  </div>
</div>
<p class="ss-sub">{subtitle}</p>
""",
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str) -> str:
    """kind: shipped | control | bench."""
    return f'<span class="ss-badge ss-badge-{kind}">{html.escape(text)}</span>'


def model_header(name: str, badge_text: str, badge_kind: str) -> None:
    st.markdown(
        f'<div class="ss-colhead"><span class="ss-model">{html.escape(name)}</span>'
        f"{badge(badge_text, badge_kind)}</div>",
        unsafe_allow_html=True,
    )


def callout(text_html: str, kind: str = "info", icon: str = "info") -> None:
    """kind: info | amber. text_html may contain markup — escape caller data."""
    extra = " ss-callout-amber" if kind == "amber" else ""
    st.markdown(
        f'<div class="ss-callout{extra}">{ICONS[icon]}<div>{text_html}</div></div>',
        unsafe_allow_html=True,
    )


def feature_card(icon_name: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="ss-feature"><div class="ss-feat-icon">{icon(icon_name, 20)}</div>'
        f'<div class="ss-feat-title">{html.escape(title)}</div>'
        f'<p class="ss-feat-desc">{html.escape(desc)}</p></div>',
        unsafe_allow_html=True,
    )


def step(num: int, text_html: str) -> None:
    """Numbered routing-step row. text_html may contain markup."""
    st.markdown(
        f'<div class="ss-step"><span class="ss-step-num">{num}</span>'
        f"<div>{text_html}</div></div>",
        unsafe_allow_html=True,
    )


def stat_tile(label: str, value: str) -> None:
    st.markdown(
        f'<div class="ss-stat"><div class="ss-stat-label">{html.escape(label)}</div>'
        f'<div class="ss-stat-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def stat_cards(cards: list[dict]) -> None:
    """Row of stat tiles: [{label, value, sub?, accent?}, ...]. Reflows on mobile."""
    parts = ['<div class="ss-cards">']
    for c in cards:
        cls = "ss-stat ss-stat-accent" if c.get("accent") else "ss-stat"
        sub = c.get("sub", "")
        parts.append(
            f'<div class="{cls}"><div class="ss-stat-label">{html.escape(c["label"])}</div>'
            f'<div class="ss-stat-value">{html.escape(str(c["value"]))}</div>'
            + (f'<div class="ss-stat-sub">{html.escape(sub)}</div>' if sub else "")
            + "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def pills(items: list[str]) -> None:
    """At-a-glance spec row (dataset size, method, test count)."""
    inner = "".join(f'<span class="ss-pill">{html.escape(t)}</span>' for t in items)
    st.markdown(f'<div class="ss-pills">{inner}</div>', unsafe_allow_html=True)


def section(title: str, icon_name: str = "layers") -> None:
    st.markdown(
        f'<div class="ss-section">{icon(icon_name, 17)}<h2>{html.escape(title)}</h2></div>',
        unsafe_allow_html=True,
    )


def id_chip(key: str, value: str, head: int = 8, tail: int = 6) -> None:
    """Show a long opaque identifier without letting it dominate the layout.

    The anonymized H&M customer_id is a 64-char hash; printed in full it is the
    widest thing on the page and says nothing. Truncated and monospaced it
    still signals "this is a real row from the dataset", and the full value
    stays available on hover for anyone who wants to check it.
    """
    short = value if len(value) <= head + tail + 1 else f"{value[:head]}…{value[-tail:]}"
    st.markdown(
        f'<span class="ss-idchip" title="{html.escape(value)}">'
        f'<span class="ss-idchip-key">{html.escape(key)}</span>{html.escape(short)}</span>',
        unsafe_allow_html=True,
    )


def col_note(text_html: str) -> None:
    """Small footnote under a model column. text_html may contain markup."""
    st.markdown(f'<div class="ss-colnote">{text_html}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        f'<div class="ss-footer">Built by Ryan Murray — '
        f'<a href="{REPO_URL}">source on GitHub</a>. '
        "Every number in this demo is reproducible from the repo: one data "
        "pipeline, one training run, a fixed seed, and a persisted "
        "train/test split.</div>",
        unsafe_allow_html=True,
    )


# H&M colour_group_name → swatch hex. Keyword heuristic (checked in order) so
# unseen groups like "Light Dusty Blue" still land on a sensible swatch.
_SWATCH_KEYWORDS = [
    ("black", "#1F1F23"),
    ("white", "#F4F4F5"),
    ("denim", "#4A6FA5"),
    ("navy", "#1E3A8A"),
    ("blue", "#3B82F6"),
    ("turquoise", "#2DD4BF"),
    ("green", "#4D7C0F"),
    ("khaki", "#8A865A"),
    ("yellow", "#EAB308"),
    ("mustard", "#CA8A04"),
    ("orange", "#F97316"),
    ("red", "#DC2626"),
    ("pink", "#EC4899"),
    ("rose", "#F43F5E"),
    ("purple", "#8B5CF6"),
    ("lilac", "#C4B5FD"),
    ("mole", "#8B7E74"),
    ("brown", "#8B5E34"),
    ("beige", "#D6C7A8"),
    ("mauve", "#B08BA5"),
    ("grey", "#9CA3AF"),
    ("gray", "#9CA3AF"),
    ("silver", "#C7CCD4"),
    ("gold", "#D4AF37"),
    ("bronze", "#B08D57"),
]


def swatch_hex(colour_group_name: str) -> str:
    name = (colour_group_name or "").lower()
    for keyword, hex_code in _SWATCH_KEYWORDS:
        if keyword in name:
            return hex_code
    return "#A1A1AA"


def product_card(item: dict, rank: int | None = None, compact: bool = False) -> str:
    """Card HTML for one article (no images in the dataset — swatch + text)."""
    name = html.escape(str(item.get("prod_name", item.get("article_id", "?"))))
    ptype = html.escape(str(item.get("product_type_name", "") or ""))
    colour = str(item.get("colour_group_name", "") or "")
    meta = " · ".join(p for p in [ptype, html.escape(colour)] if p)
    rank_html = f'<span class="ss-rank">{rank}</span>' if rank is not None else ""
    cls = "ss-card ss-card-compact" if compact else "ss-card"
    return (
        f'<div class="{cls}">{rank_html}'
        f'<span class="ss-swatch" style="background:{swatch_hex(colour)}" title="{html.escape(colour)}"></span>'
        f'<div class="ss-body"><div class="ss-name">{name}</div>'
        f'<div class="ss-meta">{meta or "&nbsp;"}</div></div></div>'
    )


def product_list(items: list[dict], ranked: bool = True, compact: bool = False, start: int = 1) -> None:
    """Render a ranked list in one markdown call. `start` keeps rank numbers
    continuous when a list is split across a fold (e.g. 1-6 then 7-12)."""
    cards = "".join(
        product_card(item, rank=start + i if ranked else None, compact=compact)
        for i, item in enumerate(items)
    )
    st.markdown(cards, unsafe_allow_html=True)
