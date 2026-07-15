"""Human-readable benchmark tables."""

from __future__ import annotations


def print_table(title: str, headers: list[tuple[str, str]], rows: list[dict]) -> None:
    print(f"\n{title}")
    labels = [label for _, label in headers]
    widths = [len(label) for label in labels]
    rendered = []
    for row in rows:
        values = []
        for index, (key, _) in enumerate(headers):
            value = row.get(key, "-")
            if isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            widths[index] = max(widths[index], len(value))
            values.append(value)
        rendered.append(values)
    print("  ".join(label.rjust(width) for label, width in zip(labels, widths)))
    print("  ".join("-" * width for width in widths))
    for values in rendered:
        print("  ".join(value.rjust(width) for value, width in zip(values, widths)))


def print_results(rows: list[dict]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    skipped = [row for row in rows if row.get("status") != "ok"]
    prefill = [row for row in ok if row["suite"] == "prefill"]
    decode = [row for row in ok if row["suite"] == "decode"]
    offline = [row for row in ok if row["suite"] == "offline"]
    online = [row for row in ok if row["suite"] == "online"]

    if prefill:
        print_table(
            "Prefill (exact prompt ingestion)",
            [
                ("batch_size", "batch"),
                ("context_tokens", "input"),
                ("wall_ms", "wall ms"),
                ("gpu_ms", "GPU ms"),
                ("prompt_tokens_per_second", "prompt tok/s"),
                ("peak_vram_gb", "VRAM GB"),
            ],
            prefill,
        )
    if decode:
        print_table(
            "Decode (prefilled cache, fixed steps)",
            [
                ("batch_size", "batch"),
                ("context_tokens", "context"),
                ("wall_ms_per_step", "ms/step"),
                ("gpu_ms_per_step", "GPU ms/step"),
                ("output_tokens_per_second", "output tok/s"),
                ("peak_vram_gb", "VRAM GB"),
            ],
            decode,
        )
    if offline:
        print_table(
            "Offline saturation (all requests ready)",
            [
                ("batch_size", "batch"),
                ("input_tokens", "input"),
                ("output_tokens", "output"),
                ("output_tokens_per_second", "output tok/s"),
                ("request_throughput", "req/s"),
                ("ttft_p50_ms", "TTFT p50"),
                ("tpot_p50_ms", "TPOT p50"),
                ("peak_vram_gb", "VRAM GB"),
            ],
            offline,
        )
    if online:
        print_table(
            "Online serving (Poisson arrivals)",
            [
                ("offered_requests_per_second", "offered req/s"),
                ("realized_offered_requests_per_second", "realized req/s"),
                ("request_throughput", "done req/s"),
                ("output_tokens_per_second", "output tok/s"),
                ("ttft_p50_ms", "TTFT p50"),
                ("ttft_p95_ms", "TTFT p95"),
                ("tpot_p50_ms", "TPOT p50"),
                ("tpot_p95_ms", "TPOT p95"),
                ("goodput_percent", "good %"),
            ],
            online,
        )
    if skipped:
        reasons: dict[str, int] = {}
        for row in skipped:
            reasons[row["status"]] = reasons.get(row["status"], 0) + 1
        summary = ", ".join(
            f"{reason}: {count}" for reason, count in sorted(reasons.items())
        )
        print(f"\nSkipped {len(skipped)} configuration(s) ({summary}).")
