from .parsing import Item


def total_value(items: list[Item]) -> float:
    """Sum of qty * price over all items."""
    return round(sum(it.qty * it.price for it in items), 2)


def median_price(items: list[Item]) -> float:
    """Median unit price; for an even number of items, the mean of the two middle prices."""
    if not items:
        raise ValueError("no items")
    prices = sorted(it.price for it in items)
    mid = len(prices) // 2
    if len(prices) % 2:
        return prices[mid]
    return prices[mid]
