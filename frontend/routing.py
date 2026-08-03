"""The hybrid's cold-start routing, described for the UI.

Mirrors ``HybridRecommender.recommend`` (src/shelfsense/hybrid.py) so every
page can tell a visitor *why* they are seeing what they are seeing — the
routing is the interesting part of the model, and it is invisible if the app
only ever prints the output.

The threshold is duplicated rather than imported: the Streamlit frontend is
deployed on its own, without the ``src/shelfsense`` package importable, so
this file must stay in step with ``config.MIN_INTERACTIONS_FOR_CF`` by hand.
"""

MIN_INTERACTIONS_FOR_CF = 3  # keep in sync with shelfsense.config

BRANCHES = [
    {
        "when": "No history at all",
        "name": "Popularity",
        "why": "0 purchases — no signal to personalize on",
        "how": (
            "Falls back to global popularity: what the whole store is buying "
            "right now. Not a placeholder — it is the hardest baseline to beat."
        ),
    },
    {
        "when": "Thin history",
        "name": "Content",
        "why": f"1–{MIN_INTERACTIONS_FOR_CF - 1} purchases — too few to trust a category",
        "how": (
            "Content-based similarity: TF-IDF over article metadata (product "
            "type, colour, department) to find items like the ones they bought."
        ),
    },
    {
        "when": "Enough history",
        "name": "Category affinity",
        "why": f"{MIN_INTERACTIONS_FOR_CF}+ purchases — a favourite category is visible",
        "how": (
            "Best-sellers inside the customer's own favourite product category, "
            "topped up with global popularity if that category is too small."
        ),
    },
]


def routing_branch(n_purchases: int | None) -> dict:
    """Which branch a customer with this much history falls down.

    ``None`` (an API too old to report purchase counts) is treated as the
    warm case, matching what the sampled customers overwhelmingly are.
    """
    if n_purchases is None:
        return BRANCHES[2]
    if n_purchases <= 0:
        return BRANCHES[0]
    if n_purchases < MIN_INTERACTIONS_FOR_CF:
        return BRANCHES[1]
    return BRANCHES[2]
