"""Explicit input order matching upstream ask-price/size, bid-price/size per level."""


def lob_columns(levels=3):
    if levels not in {3, 5}:
        raise ValueError("Only three or five LOB levels are configured")
    return tuple(f"{side}_{kind}{level}" for level in range(1, levels + 1)
                 for side, kind in (("ask", "price"), ("ask", "volume"), ("bid", "price"), ("bid", "volume")))
