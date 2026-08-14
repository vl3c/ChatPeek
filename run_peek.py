#!/usr/bin/env python3
from pathlib import Path
import time
import argparse
from ChatPeek import fetch_share_page, parse_share_html, ExportOptions
import sys

def main():
    p = argparse.ArgumentParser(description="Run ChatPeek with retries and extended timeout")
    p.add_argument("--url", "-u", required=True, help="https://chatgpt.com/share/...")
    p.add_argument("--output", "-o", default="Exports", help="Output directory")
    p.add_argument("--attempts", "-a", type=int, default=3, help="How many attempts")
    p.add_argument("--timeout", "-t", type=int, default=120, help="Fetch timeout in seconds")
    p.add_argument("--skip-assets", action="store_true", help="Do not download images/attachments")
    p.add_argument("--include-reasoning", action="store_true")
    p.add_argument("--include-tool-output", action="store_true")
    p.add_argument("--include-model-context", action="store_true")
    args = p.parse_args()

    url = args.url
    attempts = max(1, args.attempts)
    timeout = max(10, args.timeout)
    output_dir = Path(args.output)

    options = ExportOptions(
        include_reasoning=args.include_reasoning,
        include_tool_output=args.include_tool_output,
        include_model_context=args.include_model_context,
    )

    last_exc = None
    for i in range(1, attempts + 1):
        try:
            print(f"[{i}/{attempts}] Fetching share page (timeout={timeout}s)...")
            html = fetch_share_page(url, timeout=timeout)
            print("Parsing share HTML...")
            chat = parse_share_html(html, options)
            print("Saving markdown (may download assets)...")
            md_path = chat.save_markdown(output_dir, download_assets=not args.skip_assets)
            print(f"Export written: {md_path}")
            # Print first/last markers for quick check
            try:
                content = md_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                print("\n--- File head (first 30 lines) ---")
                print("\n".join(lines[:30]))
                print("\n--- File tail (last 30 lines) ---")
                print("\n".join(lines[-30:]))
            except Exception:
                pass
            return 0
        except Exception as exc:
            last_exc = exc
            print(f"Attempt {i} failed: {exc}", file=sys.stderr)
            if i < attempts:
                wait = 5 * i
                print(f"Waiting {wait}s before retry...")
                time.sleep(wait)
    print("All attempts failed.", file=sys.stderr)
    if last_exc:
        print(f"Last error: {last_exc}", file=sys.stderr)
    return 1

if __name__ == "__main__":
    sys.exit(main())
