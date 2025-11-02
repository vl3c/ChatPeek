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
python ChatPeek.py https://chatgpt.com/share/690781ed-75f0-8006-9d6e-d9229bd932f2
# -> ./gigawatt-data-centers-690781ed.md
```

From Python you can orchestrate the export yourself:

```python
from pathlib import Path
from ChatPeek import ChatPeek

peek = ChatPeek("https://chatgpt.com/share/690781ed-75f0-8006-9d6e-d9229bd932f2")
markdown_path = peek.chat.save_markdown(Path("exports"), download_assets=True)
print(markdown_path)
```

## Testing

All behaviour is covered by unit tests. They replay a single saved HTML fixture (no live traffic). Run them with:

```bash
python -m unittest ChatPeek_test.py
```

## Responsible use

ChatPeek is designed for personal conversation exports. It makes one GET request per share (matching a real browser) and uses the data already embedded in the page - there is no scraping of private APIs or bulk harvesting. Please keep usage within those boundaries.

## License

This project is licensed under the MIT License.
