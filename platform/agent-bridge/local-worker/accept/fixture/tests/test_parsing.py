from invstats import Item, parse_csv

CSV = """name,qty,price
apple, 3, 0.5
pear,2,1.25

plum,10,0.2
"""


def test_parse_csv_reads_every_row():
    items = parse_csv(CSV)
    assert [it.name for it in items] == ["apple", "pear", "plum"]
    assert items[0] == Item("apple", 3, 0.5)


def test_parse_csv_header_only():
    assert parse_csv("name,qty,price\n") == []
