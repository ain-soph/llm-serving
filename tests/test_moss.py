import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SERVER = Path(__file__).resolve().parents[1] / "services/audio/moss/server.py"


class FakeTokenizer:
    def convert_tokens_to_ids(self, tag):
        return {"<think>": 1, "</think>": 2}[tag]

    def decode(self, tokens, **kwargs):
        return "".join({3: "Plan", 4: "Answer"}[token] for token in tokens)


class MossTokenEventsTests(unittest.TestCase):
    def test_reasoning_and_answer_preserve_incremental_text(self):
        spec = importlib.util.spec_from_file_location("moss_server", SERVER)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        model = SimpleNamespace(
            _ensure_processor=lambda: SimpleNamespace(tokenizer=FakeTokenizer()),
            _generate_tokens=lambda *args, **kwargs: iter([1, 3, 2, 4, 0]),
            config=SimpleNamespace(eos_token_id=0),
        )
        self.assertEqual(list(server.token_events(model, (None, None, None), {})), [
            {"reasoning_content": "Plan"},
            {"content": "Answer"},
            ("stop", 4),
        ])


if __name__ == "__main__":
    unittest.main()
