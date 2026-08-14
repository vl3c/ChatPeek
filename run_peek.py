#!/usr/bin/env python3
"""Zero-interaction runner for ChatPeek

Run this script with no arguments. It will:
- auto-install any missing Python dependencies from requirements.txt
- fetch the provided ChatGPT share URL (default is embedded)
- parse and export the conversation to Exports/
- zip the Exports/ folder to chatpeek-exports.zip for easy download

Just run:
  python3 run_peek.py

If you want to override the URL or behaviour, optional flags are available but not required.
"""

from pathlib import Path
import sys
import subprocess
import time
import argparse
import shutil
import zipfile

# Default public share URL provided earlier
DEFAULT_SHARE_URL = "https://chatgpt.com/share/6a7de646-63e0-83ed-8696-7351947b89ad"

# Ensure requirements are installed if missing
def ensure_requirements():
    try:
        import requests  # noqa: F401
    except Exception:
        req = Path("requirements.txt")
        if req.exists():
            print("Dependencies missing: installing from requirements.txt...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
            except subprocess.CalledProcessError as e:
                print("Failed to install dependencies:", e, file=sys.stderr)
                sys.exit(2)
        else:
            print("requirements.txt not found and 'requests' missing; please install dependencies manually.", file=sys.stderr)
            sys.exit(2)


def zip_exports(exports_dir: Path, out_zip: Path):
    if not exports_dir.exists():
        print(f"No exports directory at {exports_dir}; nothing to zip.")
        return False
    if out_zip.exists():
        out_zip.unlink()
    print(f"Creating archive {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(exports_dir.rglob("*")):
            if p.is_dir():
                continue
            zf.write(p, p.relative_to(exports_dir.parent))
    print(f"Archive created: {out_zip.resolve()}")
    return True


def run_export(url: str, output: Path, attempts: int, timeout: int, skip_assets: bool, include_tool_output: bool, include_reasoning: bool, include_model_context: bool):
    # Import ChatPeek components (module is in repo)
    try:
        from ChatPeek import fetch_share_page, parse_share_html, ExportOptions
    except Exception as e:
        print("Failed to import ChatPeek module:", e, file=sys.stderr)
        print("Ensure you're running this script from the repository root where ChatPeek.py is located.")
        sys.exit(3)

    options = ExportOptions(
        include_reasoning=include_reasoning,
        include_tool_output=include_tool_output,
        include_model_context=include_model_context,
    )

    last_exc = None
    for i in range(1, attempts + 1):
        try:
            print(f"[{i}/{attempts}] Fetching share page (timeout={timeout}s)...")
            html = fetch_share_page(url, timeout=timeout)
            print("Parsing share HTML...")
            chat = parse_share_html(html, options)
            print("Saving markdown (may download assets)...")
            md_path = chat.save_markdown(output, download_assets=not skip_assets)
            print(f"Export written: {md_path}")
            return md_path
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
    sys.exit(4)


def main(argv=None):
    ensure_requirements()

    p = argparse.ArgumentParser(description="Zero-interaction ChatPeek runner")
    p.add_argument("--url", "-u", default=DEFAULT_SHARE_URL, help="ChatGPT share URL (default embedded)")
    p.add_argument("--output", "-o", default="Exports", help="Output directory")
    p.add_argument("--attempts", "-a", type=int, default=3, help="Number of fetch attempts")
    p.add_argument("--timeout", "-t", type=int, default=120, help="Fetch timeout seconds")
    p.add_argument("--skip-assets", action="store_true", help="Do not download images/attachments")
    p.add_argument("--include-reasoning", action="store_true")
    p.add_argument("--include-tool-output", action="store_true")
    p.add_argument("--include-model-context", action="store_true")
    args = p.parse_args(args=argv)

    url = args.url
    output_dir = Path(args.output)

    print("Starting ChatPeek export...\n")
    md_path = run_export(
        url,
        output_dir,
        attempts=args.attempts,
        timeout=args.timeout,
        skip_assets=args.skip_assets,
        include_tool_output=args.include_tool_output,
        include_reasoning=args.include_reasoning,
        include_model_context=args.include_model_context,
    )

    # Zip Exports for easy download
    out_zip = Path("chatpeek-exports.zip")
    zipped = zip_exports(output_dir, out_zip)

    print("\nDone.")
    if md_path:
        print(f"Markdown: {md_path.resolve()}")
    if zipped:
        print(f"Download archive: {out_zip.resolve()}")
    else:
        print("No archive created; download the Exports/ folder via VS Code Explorer.")


if __name__ == "__main__":
    main()
