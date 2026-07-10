"""Liveness checks for the ChatGPT share links behind the test fixtures.

The unit tests replay saved HTML and never touch the network, so they keep
passing after a share link expires. A dead link still matters though: it
breaks the README examples and, more importantly, stops anyone from
re-capturing the fixture. This module fetches each backing link and, when one
has gone stale, prints a loud, self-contained warning explaining how to mint a
replacement -- including a ready-to-paste prompt that makes an AI chat produce
a conversation with the varied content the tests need.

Usage (network required)::

    python check_fixture_links.py                    # check every backing link
    python check_fixture_links.py --capture <url>    # save a fresh fixture

`check` exits non-zero if any link is stale, so it can be wired into CI or a
scheduled job (see .github/workflows/ci.yml). The unit suite runs the same
check only when CHATPEEK_CHECK_LINKS is set, to keep the default run offline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ChatPeek import ShareAccessError, fetch_share_page, parse_share_html

FIXTURES: Path = Path(__file__).resolve().parent / "fixtures"

# Prompt a maintainer can paste into a fresh AI chat to regenerate a
# conversation with the spread of content the exporter tests exercise. Keep it
# in sync with the assertions in VariedShareFixtureTests.
REGENERATION_PROMPT: str = """\
I'm building test fixtures for a tool that exports ChatGPT share links to
Markdown, and I need a conversation with deliberately varied content. Please
produce a multi-turn exchange that includes ALL of the following, so the
exporter gets exercised. Keep every topic neutral and timeless -- no current
events, no personal data:

1. At least six alternating user/assistant turns.
2. Fenced code blocks in several languages: C++ (include the exact line
   `#include <iostream>`), HTML, and Python.
3. A fenced ```md code block whose *contents* are themselves Markdown -- a
   line `# Heading 1`, a nested bullet list, and a small table -- so
   nested-Markdown handling can be verified.
4. At least one ordinary Markdown table rendered in a normal answer.
5. Prose paragraphs, bullet lists, and inline **bold** / *italic* / `code`.

When you're done I'll use the Share button to create a public link."""


@dataclass
class FixtureLink:
    """A saved fixture and the live share link it was captured from."""

    fixture: str
    url: str
    # Substrings that must survive a round-trip through the exporter for the
    # live conversation to still be a usable source for this fixture.
    content_markers: List[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return FIXTURES / self.fixture


FIXTURE_LINKS: List[FixtureLink] = [
    FixtureLink(
        fixture="69b1c492-1540-8006-aa29-ee2e0a831385.html",
        url="https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385",
        content_markers=["#include <iostream>", "# Heading 1", "```html"],
    ),
]


def check_fixture_link(link: FixtureLink) -> Optional[str]:
    """Return None if the link is healthy, else a maintainer-facing warning."""

    try:
        html = fetch_share_page(link.url)
    except ShareAccessError as exc:
        return _stale_message(link, f"the share link is no longer public ({exc})")
    except Exception as exc:  # any network/HTTP failure means the link is unusable
        return _stale_message(link, f"the share link could not be fetched ({exc})")

    try:
        markdown = parse_share_html(html).to_markdown()
    except Exception as exc:  # a page that no longer parses is just as stale
        return _stale_message(link, f"the share page no longer parses ({exc})")

    missing = [marker for marker in link.content_markers if marker not in markdown]
    if missing:
        joined = ", ".join(repr(marker) for marker in missing)
        return _stale_message(
            link,
            "the link is live but its content changed; the exporter no longer "
            f"produces: {joined}",
        )
    return None


def _stale_message(link: FixtureLink, reason: str) -> str:
    bar = "=" * 72
    return "\n".join(
        [
            bar,
            f"STALE FIXTURE LINK: {link.fixture}",
            bar,
            f"URL: {link.url}",
            f"Problem: {reason}.",
            "",
            "The saved fixture still works, but this link can no longer source a",
            "fresh copy of it. To provide a replacement:",
            "",
            "  1. Open a fresh ChatGPT session and paste the prompt below.",
            "  2. Let it answer, then use Share -> create link and copy the public",
            "     https://chatgpt.com/share/... URL.",
            "  3. Capture it as a fixture:",
            "",
            "       python check_fixture_links.py --capture <NEW_SHARE_URL>",
            "",
            "  4. Point FIXTURE_LINKS (this file), the README, and any tests that",
            "     name the old share id at the new fixture, then run the suite.",
            "",
            "----- PROMPT TO PASTE INTO THE AI CHAT -----",
            REGENERATION_PROMPT,
            "--------------------------------------------",
            bar,
        ]
    )


def capture(url: str) -> Path:
    """Fetch ``url`` and save it under fixtures/ named after its share id."""

    share_id = url.rstrip("/").split("/")[-1]
    if not share_id:
        raise ValueError(f"Could not derive a share id from URL: {url!r}")
    html = fetch_share_page(url)
    out = FIXTURES / f"{share_id}.html"
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the share links behind the test fixtures are still live."
    )
    parser.add_argument(
        "--capture",
        metavar="SHARE_URL",
        help="Fetch SHARE_URL and save it under fixtures/ as <share-id>.html",
    )
    args = parser.parse_args(argv)

    if args.capture:
        out = capture(args.capture)
        print(f"Saved {out}")
        return 0

    stale: List[str] = []
    for link in FIXTURE_LINKS:
        message = check_fixture_link(link)
        if message is None:
            print(f"OK: {link.fixture} <- {link.url}")
        else:
            print(message, file=sys.stderr)
            stale.append(link.fixture)

    if stale:
        print(
            f"\n{len(stale)} stale fixture link(s): {', '.join(stale)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
