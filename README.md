# ChatPeek - export ChatGPT shares to Markdown

ChatPeek downloads an individual `chatgpt.com/share/...` link (using the same headers a private browser tab would send), rebuilds the conversation offline, and turns it into clean Markdown. When the conversation points to downloadable assets (images, attachments), the exporter can create a conversation folder containing the Markdown plus the referenced files.

## Features

- Single-request fetch with private-window style headers (no retry loop or backend probing)
- React Flight parser for modern `chatgpt.com` shares with a legacy fallback for older `chat.openai.com` pages
- Message normalisation that keeps Markdown tables, code blocks, thoughts, tool outputs, and attachment links
- Markdown writer with optional asset download (`images/` and `attachments/` subfolders when needed)
- Lightweight CLI: `python ChatPeek.py <share-url>` writes the Markdown to disk

## Quick start

```bash
python -m pip install -r requirements.txt  # requests + beautifulsoup4
python ChatPeek.py https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385
# -> ./hybrid-cognitive-systems-69b1c492.md
```

The share fetch retries transient failures - connection errors, timeouts, and HTTP 429/5xx - up to three times with a linear backoff. A deterministic answer such as a deleted share or a private conversation fails immediately instead, since retrying it would return the same result. Pass `--attempts 1` to disable retrying, or a higher count on an unreliable connection.

By default, ChatPeek now omits internal reasoning traces, tool outputs, and model editable context from the export. If you want to include them, use:

```bash
python ChatPeek.py https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385 \
  --include-reasoning \
  --include-tool-output \
  --include-model-context
```

From Python you can orchestrate the export yourself:

```python
from pathlib import Path
from ChatPeek import ChatPeek

peek = ChatPeek("https://chatgpt.com/share/69b1c492-1540-8006-aa29-ee2e0a831385")
markdown_path = peek.chat.save_markdown(Path("exports"), download_assets=True)
print(markdown_path)
```

## Testing

All behaviour is covered by unit tests. They replay saved HTML fixtures under `fixtures/` (no live traffic), so they keep passing even after a share link expires. Run them with:

```bash
python -m unittest ChatPeek_test.py
```

### Keeping the fixture links fresh

The fixtures were captured from real share links, and those links can go dead over time. A separate check confirms the backing links are still live and still expose the varied content the tests rely on:

```bash
python check_fixture_links.py        # exits non-zero and prints guidance if any link is stale
```

A link that is genuinely gone (or whose content changed) is reported as stale and fails the check. A transient problem — a network error, rate limit, or bot challenge, which datacenter CI runners hitting ChatGPT sometimes get — is reported as inconclusive and does *not* fail, so the weekly check does not raise false alarms.

If a link is stale, the output tells you how to mint a replacement — including a ready-to-paste prompt that makes an AI chat generate a suitably varied conversation — and how to re-capture the fixture:

```bash
python check_fixture_links.py --capture https://chatgpt.com/share/<new-id>
```

CI runs the offline tests on every push and pull request, and runs the liveness check on a weekly schedule so a stale link surfaces on its own (a failed scheduled run notifies the repository owner). You can also set `CHATPEEK_CHECK_LINKS=1` to fold the liveness check into the unit-test run locally.

## Responsible use

ChatPeek is designed for personal conversation exports. It makes one GET request per share (matching a real browser) and uses the data already embedded in the page - there is no scraping of private APIs or bulk harvesting. Please keep usage within those boundaries.

## License

This project is licensed under the MIT License.
