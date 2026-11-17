"""Check a running reranker against public fixtures and optionally benchmark it.

Reserve exclusive GPU access before using --benchmark. The server must be started
separately; this tool neither loads models nor changes system services.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18189")
    parser.add_argument("--size", choices=["0.6b", "4b", "8b"], required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    fixture = json.loads((ROOT / "tests/fixtures/rerank-reference.json").read_text())
    model_id = "qwen3-reranker-" + args.size
    assert request(args.url, "/health")["status"] == "ok"
    assert request(args.url, "/v1/models")["data"][0]["id"] == model_id
    worst = 0.0
    for case, expected in zip(fixture["cases"], fixture["official_bf16_scores"][args.size], strict=True):
        response = request(args.url, "/v1/rerank", case)
        assert response["model"] == model_id
        results = response["results"]
        assert len(results) == len(expected)
        scores = [None] * len(expected)
        for result in results:
            scores[result["index"]] = result["relevance_score"]
        assert all(x is not None and 0 <= x <= 1 for x in scores)
        assert [x["index"] for x in results] == sorted(range(len(expected)), key=lambda i: -expected[i])
        worst = max(worst, max(abs(x-y) for x, y in zip(scores, expected, strict=True)))
    assert worst < {"0.6b": 0.002, "4b": 0.012, "8b": 0.006}[args.size], worst
    case = fixture["cases"][0]
    response = request(args.url, "/rerank", {**case, "documents": [{"text": d} for d in case["documents"]],
                                           "top_n": 1, "return_documents": True})
    assert len(response["results"]) == 1
    assert response["results"][0]["document"]["text"] == case["documents"][response["results"][0]["index"]]
    assert request(args.url, "/v1/reranking", case)["results"][0]["index"] == response["results"][0]["index"]
    try:
        request(args.url, "/v1/rerank", {"query": "test", "documents": []})
    except urllib.error.HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Empty documents must return HTTP 400")
    result = {"model": model_id, "reference_pairs": 11, "same_rankings": True,
              "max_absolute_score_error": worst, "api_checks": "passed"}
    if args.benchmark:
        para = ("Apple Silicon uses unified memory shared by the CPU and GPU. Metal exposes it to compute kernels, and "
                "llama.cpp maps model weights into that memory so that inference avoids copies. ")
        body = {"query": "How does unified memory help local LLM inference on a Mac?",
                "documents": [f"Document {i}. " + para * 12 for i in range(20)]}
        request(args.url, "/v1/rerank", body)  # warm-up, excluded
        times = []
        for _ in range(3):
            start = time.perf_counter()
            response = request(args.url, "/v1/rerank", body)
            times.append(time.perf_counter() - start)
            assert len(response["results"]) == 20
        mean = statistics.mean(times)
        result["benchmark"] = {"documents": 20, "concurrency": 1, "warmup_requests": 1,
                               "latency_samples_seconds": times, "mean_seconds": mean,
                               "median_seconds": statistics.median(times),
                               "prompt_tokens": response["usage"]["prompt_tokens"],
                               "input_tokens_per_second": response["usage"]["prompt_tokens"] / mean,
                               "measurement": "HTTP end-to-end input tokens/s, not isolated kernel prefill"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
