"""Hidden checks run by the verifier (not shown to the worker): python -m pytest -q <this file> from the repo root."""
import pytest

from invstats import Item, median_price, parse_csv, top_n, total_value
import invstats


def test_exports():
    assert "top_n" in invstats.__all__


def test_parse_many_rows_and_blank_lines():
    text = "name,qty,price\n\nx,1,1\n y , 2 , 2.5 \n\nz,3,0\n"
    assert [(i.name, i.qty, i.price) for i in parse_csv(text)] == [("x", 1, 1.0), ("y", 2, 2.5), ("z", 3, 0.0)]


def test_median_even_unsorted():
    items = [Item("a", 1, 9.0), Item("b", 1, 1.0), Item("c", 1, 3.0), Item("d", 1, 5.0)]
    assert median_price(items) == pytest.approx(4.0)


def test_top_n_all_and_more_than_len():
    items = [Item("a", 1, 1.0), Item("b", 5, 1.0), Item("c", 2, 1.0)]
    assert [i.name for i in top_n(items, 10)] == ["b", "c", "a"]
    assert [i.name for i in top_n(items, 1)] == ["b"]


def test_total_value_unchanged():
    assert total_value([Item("a", 3, 0.5), Item("b", 2, 1.25)]) == 4.0
