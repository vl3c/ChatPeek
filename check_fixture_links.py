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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from ChatPeek import ShareAccessError, fetch_share_page, parse_share_html

FIXTURES: Path = Path(__file__).resolve().parent / "fixtures"

# HTTP statuses that mean "we were blocked or the server hiccuped", not "the
# share is gone". Datacenter IPs (e.g. CI runners) hitting chatgpt.com behind
# Cloudflare routinely get these, so they must not be reported as stale.
_TRANSIENT_STATUSES = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
# Statuses that genuinely mean the share no longer exists.
_GONE_STATUSES = frozenset({404, 410})
# A share id is a URL path segment; keep it to filesystem-safe characters.
_SHARE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

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


# Only links we still expect to be live belong here. The other fixture,
# 690781ed-...html, was captured from a share link that is no longer public;
# it is kept purely as a frozen snapshot for the tests, so there is nothing to
# re-source and it is deliberately not liveness-checked.
FIXTURE_LINKS: List[FixtureLink] = [
    FixtureLink(
        fixture="69b1c492-1540-8006-aa29-ee2e0a831385.html",
        url="https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385",
        content_markers=["#include <iostream>", "# Heading 1", "```html"],
    ),
]


@dataclass
class CheckResult:
    """Outcome of checking one link.

    ``stale`` means the share is genuinely gone or its content changed and the
    fixture should be re-sourced. ``inconclusive`` means we could not tell
    (network error, rate limit, bot challenge) — callers should surface it but
    NOT treat it as a failure, so a blocked CI runner never raises a false
    alarm. ``ok`` means the link still reproduces the fixture's content.
    """

    fixture: str
    status: str  # "ok" | "stale" | "inconclusive"
    message: Optional[str] = None


def check_fixture_link(link: FixtureLink) -> CheckResult:
    """Fetch the link and classify it as ok / stale / inconclusive."""

    try:
        html = fetch_share_page(link.url)
    except ShareAccessError as exc:
        return CheckResult(
            link.fixture, "stale", _stale_message(link, f"the share link is no longer public ({exc})")
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in _GONE_STATUSES:
            return CheckResult(
                link.fixture, "stale", _stale_message(link, f"the share link returned HTTP {status} (gone)")
            )
        return CheckResult(
            link.fixture, "inconclusive", _inconclusive_message(link, f"HTTP {status} (blocked or transient)")
        )
    except requests.RequestException as exc:
        return CheckResult(
            link.fixture, "inconclusive", _inconclusive_message(link, f"could not be reached ({exc})")
        )

    try:
        markdown = parse_share_html(html).to_markdown()
    except Exception as exc:
        # A 200 that does not parse is far more likely an interstitial or bot
        # challenge than a genuinely changed conversation, so do not cry wolf.
        return CheckResult(
            link.fixture, "inconclusive", _inconclusive_message(link, f"returned an unparseable page ({exc})")
        )

    missing = [marker for marker in link.content_markers if marker not in markdown]
    if missing:
        joined = ", ".join(repr(marker) for marker in missing)
        return CheckResult(
            link.fixture,
            "stale",
            _stale_message(
                link,
                f"the exporter no longer produces {joined} from this link "
                "(the conversation changed, or ChatPeek's rendering did)",
            ),
        )
    return CheckResult(link.fixture, "ok")


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


def _inconclusive_message(link: FixtureLink, reason: str) -> str:
    return (
        f"COULD NOT CHECK {link.fixture}: {reason}. "
        f"URL: {link.url}. Not treated as stale; will retry next run."
    )


def _share_id_from_url(url: str) -> str:
    """Extract the share id (last path segment), ignoring query and fragment."""

    segment = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    if not _SHARE_ID_RE.fullmatch(segment):
        raise ValueError(f"Could not derive a filesystem-safe share id from URL: {url!r}")
    return segment


def capture(url: str) -> Path:
    """Fetch ``url`` and save it under fixtures/ named after its share id."""

    out = FIXTURES / f"{_share_id_from_url(url)}.html"
    out.write_text(fetch_share_page(url), encoding="utf-8")
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
    inconclusive: List[str] = []
    for link in FIXTURE_LINKS:
        result = check_fixture_link(link)
        if result.status == "ok":
            print(f"OK: {link.fixture} <- {link.url}")
        elif result.status == "inconclusive":
            print(result.message, file=sys.stderr)
            inconclusive.append(link.fixture)
        else:
            print(result.message, file=sys.stderr)
            stale.append(link.fixture)

    if inconclusive:
        print(
            f"\n{len(inconclusive)} link(s) could not be checked "
            f"(network/blocked, not failing): {', '.join(inconclusive)}",
            file=sys.stderr,
        )
    if stale:
        print(
            f"\n{len(stale)} stale fixture link(s): {', '.join(stale)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
