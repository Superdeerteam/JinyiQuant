"""Explicit input order matching upstream ask-price/size, bid-price/size per level."""

TRADE_COLUMNS = ("volume", "amount", "deal_number")


def lob_columns(levels=3):
    if levels not in {3, 5}:
        raise ValueError("Only three or five LOB levels are configured")
    return tuple(f"{side}_{kind}{level}" for level in range(1, levels + 1)
                 for side, kind in (("ask", "price"), ("ask", "volume"), ("bid", "price"), ("bid", "volume")))


def lob_trade_columns():
    return lob_columns(3) + TRADE_COLUMNS


def validate_input_schema(fields, book_representation="raw", trade_features="zero"):
    fields = tuple(fields)
    if fields not in (lob_columns(3), lob_columns(5), lob_trade_columns()):
        raise ValueError("Unexpected LOB field order")
    if book_representation not in ("raw", "relative_mid_bps") or trade_features not in ("zero", "log1p"):
        raise ValueError("Unsupported LOB input transformation")
    if trade_features != "zero" and fields != lob_trade_columns():
        raise ValueError("Trade features require three LOB levels followed by volume, amount, deal_number")
    return 12 if fields == lob_trade_columns() else len(fields)
