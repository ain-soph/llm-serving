import pytest

from invstats import Item, median_price, top_n, total_value

ITEMS = [Item("a", 3, 0.5), Item("b", 2, 1.25), Item("c", 10, 0.2), Item("d", 1, 4.0)]


def test_total_value():
    assert total_value(ITEMS) == 10.0


def test_median_price_odd():
    assert median_price(ITEMS[:3]) == 0.5


def test_median_price_even():
    assert median_price(ITEMS) == pytest.approx(0.875)


def test_median_price_empty():
    with pytest.raises(ValueError):
        median_price([])


def test_top_n_by_value():
    # value = qty * price: a 1.5, b 2.5, c 2.0, d 4.0
    assert [it.name for it in top_n(ITEMS, 2)] == ["d", "b"]
    assert top_n(ITEMS, 0) == []
