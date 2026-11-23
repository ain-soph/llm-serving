from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    name: str
    qty: int
    price: float


def parse_line(line: str) -> Item:
    name, qty, price = (part.strip() for part in line.split(","))
    return Item(name=name, qty=int(qty), price=float(price))


def parse_csv(text: str) -> list[Item]:
    """Parse CSV text whose first line is the header `name,qty,price`. Blank lines are ignored."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return [parse_line(ln) for ln in lines[2:]]
