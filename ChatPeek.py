"""Utilities for exporting ChatGPT shared conversations to Markdown."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests  # type: ignore[import]


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ),
    "Sec-Ch-Ua": '"Chromium";v="118", "Not=A?Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


EXPORT_ROOT = Path("Exports")


class _ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._current_attrs: Dict[str, str] = {}
        self._current_data: List[str] = []
        self.scripts: List[Tuple[Dict[str, str], str]] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() == "script":
            self._in_script = True
            self._current_attrs = {name: (value or "") for name, value in attrs}
            self._current_data = []

    def handle_endtag(self, tag: str):
        if tag.lower() == "script" and self._in_script:
            content = "".join(self._current_data)
            self.scripts.append((self._current_attrs, content))
            self._in_script = False
            self._current_attrs = {}
            self._current_data = []

    def handle_data(self, data: str):
        if self._in_script:
            self._current_data.append(data)


def _extract_scripts(html: str) -> List[Tuple[Dict[str, str], str]]:
    parser = _ScriptCollector()
    parser.feed(html)
    parser.close()
    return parser.scripts


class ReplyType(Enum):
    HUMAN = "user"
    AI = "assistant"
    TOOL = "tool"


@dataclass
class ConversationAsset:
    """Represents an external asset referenced by a message."""

    asset_type: str
    url: str
    filename: str
    description: Optional[str] = None
    downloadable: bool = True


@dataclass
class Reply:
    """A single message in the conversation."""

    author_name: str
    type: ReplyType
    statement: str
    created_at: Optional[float] = None
    assets: List[ConversationAsset] = field(default_factory=list)


@dataclass
class Chat:
    """Structured representation of a shared ChatGPT conversation."""

    share_id: str
    ai_model: str
    title: str
    updated_at: Optional[float]
    replies: List[Reply]

    def to_markdown(self) -> str:
        """Render the conversation as a Markdown string."""

        header_lines = [f"# {self.title or 'ChatGPT conversation'}"]
        meta_bits = []
        if self.updated_at:
            meta_bits.append(
                datetime.fromtimestamp(self.updated_at).strftime("%Y-%m-%d %H:%M:%S")
            )
        if self.ai_model:
            meta_bits.append(f"Model: {self.ai_model}")
        if meta_bits:
            header_lines.append("_" + " | ".join(meta_bits) + "_")
        header_lines.append("")

        for reply in self.replies:
            speaker = reply.author_name or reply.type.value.title()
            header_lines.append(f"### {speaker}")
            header_lines.append(reply.statement.strip())
            header_lines.append("")

        return "\n".join(line.rstrip() for line in header_lines).rstrip() + "\n"

    def save_markdown(
        self,
        output_dir: Path,
        download_assets: bool = True,
        http_get: Optional[Callable[[str], requests.Response]] = None,
    ) -> Path:
        """Write Markdown (and optional assets) to disk.

        Args:
            output_dir: The directory where the Markdown (and optional assets) will
                be stored.
            download_assets: Whether referenced assets (images, files) should be
                downloaded. If False, the Markdown will include placeholders only.
            http_get: Optional injector for network requests (facilitates testing).

        Returns:
            The path to the Markdown file that was written.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        needs_folder = any(reply.assets for reply in self.replies)
        slug = slugify_title(self.title, self.share_id)
        base_dir = output_dir / slug if needs_folder else output_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = base_dir / f"{slug}.md"
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")

        if download_assets and needs_folder:
            images_dir = base_dir / "images"
            files_dir = base_dir / "attachments"
            for reply in self.replies:
                for asset in reply.assets:
                    target_dir = images_dir if asset.asset_type == "image" else files_dir
                    target_dir.mkdir(exist_ok=True)
                    target_path = target_dir / asset.filename
                    if target_path.exists():
                        continue
                    if not asset.downloadable or not asset.url or not asset.url.lower().startswith("http"):
                        continue
                    fetch = http_get or default_http_get
                    resp = fetch(asset.url)
                    resp.raise_for_status()
                    target_path.write_bytes(resp.content)

        return markdown_path


class ShareAccessError(RuntimeError):
    """Raised when a share URL cannot be fetched due to access restrictions."""


def default_http_get(url: str) -> requests.Response:
    return requests.get(url, headers=DEFAULT_HEADERS, timeout=30)


def fetch_share_page(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> str:
    """Fetch the shared conversation HTML once using private-window style headers."""

    merged_headers = {**DEFAULT_HEADERS, "Referer": "https://chatgpt.com/"}
    if headers:
        merged_headers.update(headers)
    response = requests.get(url, headers=merged_headers, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        parsed = urlparse(url)
        path = parsed.path or ""
        if response.status_code == 403 and parsed.netloc.endswith("chatgpt.com") and path.startswith("/c/"):
            raise ShareAccessError(
                "The provided link appears to be a private conversation. "
                "Open it while logged in and copy the public https://chatgpt.com/share/... link instead."
            ) from exc
        raise
    return response.text


def extract_loader_payload(html: str) -> Optional[List]:
    """Extract the React Flight loader payload if present."""

    for _attrs, text in _extract_scripts(html):
        if not text or "streamController.enqueue" not in text:
            continue
        start = 0
        while True:
            anchor = text.find("streamController.enqueue(", start)
            if anchor == -1:
                break
            anchor += len("streamController.enqueue(")
            end = text.find(");", anchor)
            if end == -1:
                break
            chunk = text[anchor:end].strip()
            if chunk.startswith("(") and chunk.endswith(")"):
                chunk = chunk[1:-1].strip()
            if chunk.startswith("\"") and chunk.endswith("\""):
                try:
                    chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    pass
            chunk = chunk.strip()
            if chunk.startswith("["):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    pass
            start = end + 2
    return None


def decode_loader(loader: List) -> Dict:
    """Decode the flattened loader list into dictionaries and lists."""

    cache: Dict[int, object] = {}

    def decode_key(raw_key: str) -> str:
        if isinstance(raw_key, str) and raw_key.startswith("_") and raw_key[1:].isdigit():
            idx = int(raw_key[1:])
            if 0 <= idx < len(loader) and isinstance(loader[idx], str):
                return loader[idx]
        return raw_key

    def resolve(value):
        if type(value) is int:
            if value in cache:
                return cache[value]
            if not (0 <= value < len(loader)):
                return value
            cache[value] = None
            cache[value] = resolve(loader[value])
            return cache[value]
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {decode_key(k): resolve(v) for k, v in value.items()}
        return value

    resolved: Dict[str, object] = {}
    iterator = iter(loader[1:])
    for key in iterator:
        try:
            value = next(iterator)
        except StopIteration:
            break
        if isinstance(key, str) and key not in resolved:
            resolved[key] = resolve(value)
    return resolved


def parse_modern_share(html: str) -> Chat:
    loader = extract_loader_payload(html)
    if not loader:
        raise ValueError("Modern share payload not found")

    decoded = decode_loader(loader)
    route = decoded["loaderData"]["routes/share.$shareId.($action)"]
    data = route["serverResponse"]["data"]
    share_id = route["sharedConversationId"]
    model_slug = data.get("model", {}).get("slug", "")
    title = data.get("title", "")
    updated_at = data.get("update_time")
    mapping = data.get("mapping", {})
    sequence = data.get("linear_conversation", [])

    replies: List[Reply] = []
    for entry in sequence:
        node = mapping.get(entry.get("id"))
        if not node:
            continue
        message = node.get("message")
        if not message:
            continue
        role = message.get("author", {}).get("role")
        if role == "system":
            continue
        content = message.get("content") or {}
        statement, assets = flatten_message_content(message.get("id"), content, message)
        if not statement and not assets:
            continue
        reply_type = ReplyType(role) if role in ReplyType._value2member_map_ else ReplyType.AI
        author = author_name_for_role(role)
        replies.append(
            Reply(
                author_name=author,
                type=reply_type,
                statement=statement,
                created_at=message.get("create_time"),
                assets=assets,
            )
        )

    return Chat(share_id=share_id, ai_model=model_slug, title=title, updated_at=updated_at, replies=replies)


def parse_legacy_share(html: str) -> Chat:
    script_content: Optional[str] = None
    for attrs, text in _extract_scripts(html):
        if attrs.get("id") == "__NEXT_DATA__":
            script_content = text
            break

    if not script_content:
        raise ValueError("Legacy share payload not found")
    payload = json.loads(script_content)
    data = payload["props"]["pageProps"]["serverResponse"]["data"]
    share_id = data.get("conversation_id", "shared")
    model_slug = data.get("model", {}).get("slug", "")
    title = data.get("title", "")
    updated_at = data.get("update_time")
    author_name = data.get("author_name", "User")
    replies: List[Reply] = []
    for node in data.get("linear_conversation", []):
        message = node.get("message")
        if not message:
            continue
        role = message.get("author", {}).get("role")
        if role == "system":
            continue
        content = message.get("content") or {}
        statement, assets = flatten_message_content(message.get("id"), content, message)
        if not statement and not assets:
            continue
        author = author_name if role == "user" else author_name_for_role(role)
        reply_type = ReplyType(role) if role in ReplyType._value2member_map_ else ReplyType.AI
        replies.append(
            Reply(
                author_name=author,
                type=reply_type,
                statement=statement,
                created_at=message.get("create_time"),
                assets=assets,
            )
        )

    return Chat(share_id=share_id, ai_model=model_slug, title=title, updated_at=updated_at, replies=replies)


def parse_share_html(html: str) -> Chat:
    try:
        return parse_modern_share(html)
    except (ValueError, KeyError):
        return parse_legacy_share(html)


def author_name_for_role(role: Optional[str]) -> str:
    if role == "user":
        return "User"
    if role == "tool":
        return "Tool"
    return "Assistant"


PRIVATE_USE_PATTERN = re.compile("[\uE000-\uF8FF]")


def strip_private_use(text: str) -> str:
    return PRIVATE_USE_PATTERN.sub("", text)


def flatten_message_content(
    message_id: Optional[str],
    content: Dict,
    message: Dict,
) -> Tuple[str, List[ConversationAsset]]:
    content_type = content.get("content_type")
    assets: List[ConversationAsset] = []

    def render_asset_reference(asset: ConversationAsset) -> str:
        relative_dir = "images" if asset.asset_type == "image" else "attachments"
        rel_path = Path(relative_dir) / asset.filename
        if asset.downloadable:
            if asset.asset_type == "image":
                return f"![{asset.filename}]({rel_path.as_posix()})"
            return f"[{asset.filename}]({rel_path.as_posix()})"
        label = asset.description or asset.filename
        source = asset.url or "unavailable source"
        kind = "Image" if asset.asset_type == "image" else "Attachment"
        return f"*{kind} '{label}' not included in export (source: {source}).*"

    def finalize(text: str) -> Tuple[str, List[ConversationAsset]]:
        metadata = message.get("metadata") or {}
        attachment_lines: List[str] = []

        for attachment in metadata.get("attachments", []):
            url = attachment.get("download_url") or attachment.get("file_url")
            if not url:
                continue
            mime = attachment.get("mime_type")
            filename = attachment.get("name") or build_asset_filename(message_id, len(assets), mime)
            asset_type = attachment.get("file_type") or attachment.get("type") or "file"
            downloadable = bool(url and url.lower().startswith("http"))
            assets.append(
                ConversationAsset(
                    asset_type="image" if "image" in (asset_type or "").lower() else "file",
                    url=url,
                    filename=filename,
                    description=attachment.get("title") or attachment.get("name"),
                    downloadable=downloadable,
                )
            )
            attachment_lines.append(render_asset_reference(assets[-1]))
        combined = text.strip()
        if attachment_lines:
            combined = (combined + "\n\n" if combined else "") + "\n".join(attachment_lines)
        return combined, assets

    if content_type == "text":
        parsed_parts: List[str] = []
        for part in content.get("parts", []):
            if not part:
                continue
            cleaned = strip_private_use(part).strip("\n")
            parsed = cleaned
            if cleaned.startswith("{") and cleaned.endswith("}"):
                try:
                    maybe_json = json.loads(cleaned)
                except json.JSONDecodeError:
                    maybe_json = None
                if isinstance(maybe_json, dict):
                    response = maybe_json.get("response")
                    if isinstance(response, str):
                        parsed = response
                    else:
                        parsed = maybe_json.get("content") or cleaned
            parsed_parts.append(parsed)
        parts = parsed_parts
        return finalize("\n\n".join(part for part in parts if part))

    if content_type == "code":
        language = content.get("language")
        code_text = content.get("text", "")
        lang = language if language and language != "unknown" else ""
        body = code_text.rstrip("\n")
        return finalize(f"```{lang}\n{body}\n```")

    if content_type == "thoughts":
        thoughts = []
        for thought in content.get("thoughts", []):
            summary = thought.get("summary")
            detail = thought.get("content")
            combined = ": ".join(filter(None, [summary, detail]))
            if combined:
                thoughts.append(f"_{combined}_")
        return finalize("\n\n".join(thoughts))

    if content_type == "reasoning_recap":
        recap = content.get("content", "")
        return finalize(f"_{recap.strip()}_" if recap else "")

    if content_type == "model_editable_context":
        return finalize((content.get("model_set_context", "") or "").strip())

    if content_type == "multimodal_text":
        segments: List[str] = []
        for part in content.get("parts", []):
            if isinstance(part, str):
                segments.append(strip_private_use(part))
            elif isinstance(part, dict):
                p_type = part.get("content_type") or part.get("type")
                if p_type == "text":
                    texts = part.get("text")
                    if isinstance(texts, list):
                        segments.extend(strip_private_use(t) for t in texts)
                    elif isinstance(texts, str):
                        segments.append(strip_private_use(texts))
                elif p_type in {"image_asset_pointer", "file"}:
                    pointer_raw = part.get("asset_pointer")
                    if not isinstance(pointer_raw, str) or not pointer_raw:
                        continue
                    filename = build_asset_filename(message_id, len(assets), part.get("mime_type"))
                    asset_type = "image" if "image" in (p_type or "").lower() else "file"
                    pointer = pointer_raw
                    downloadable = pointer.lower().startswith("http")
                    assets.append(
                        ConversationAsset(
                            asset_type=asset_type,
                            url=pointer,
                            filename=filename,
                            downloadable=downloadable,
                        )
                    )
                    segments.append(render_asset_reference(assets[-1]))
        return finalize("\n\n".join(segment.strip() for segment in segments if segment.strip()))

    if content_type == "tool_response":
        output = content.get("output", "")
        return finalize(strip_private_use(output))

    # Attempt a generic text conversion as last resort
    if "parts" in content:
        parts = [strip_private_use(str(part)) for part in content.get("parts", []) if part]
        return finalize("\n\n".join(parts).strip())

    return finalize("")


def build_asset_filename(message_id: Optional[str], index: int, mime_type: Optional[str]) -> str:
    base = (message_id or "asset").split("-")[0]
    extension = ""
    if mime_type and "/" in mime_type:
        extension = mime_type.split("/")[-1]
    extension = extension or "bin"
    return f"{base}-{index}.{extension}"


def slugify_title(title: str, share_id: str) -> str:
    slug_base = re.sub(r"[^a-z0-9]+", "-", (title or "chat").lower()).strip("-") or "chat"
    slug_base = slug_base[:60].rstrip("-")
    return f"{slug_base}-{share_id[:8]}"


class ChatPeek:
    """High-level facade for downloading and exporting shared conversations."""

    def __init__(self, link: str):
        self._link = link
        html = fetch_share_page(link)
        self._chat = parse_share_html(html)

    @property
    def chat(self) -> Chat:
        return self._chat


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export a ChatGPT share link to Markdown")
    parser.add_argument("share_url", help="The https://chatgpt.com/share/... link to export")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_ROOT,
        help="Destination directory for the exported conversation",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Do not download linked assets (images, attachments)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        html = fetch_share_page(args.share_url)
    except ShareAccessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    chat = parse_share_html(html)
    markdown_path = chat.save_markdown(args.output, download_assets=not args.skip_assets)
    print(markdown_path)


if __name__ == "__main__":
    main()
