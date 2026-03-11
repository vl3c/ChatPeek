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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast
from urllib.parse import urlparse

import requests

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


DEFAULT_HEADERS: Dict[str, str] = {
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


EXPORT_ROOT: Path = Path("Exports")


class _ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._current_attrs: Dict[str, str] = {}
        self._current_data: List[str] = []
        self.scripts: List[Tuple[Dict[str, str], str]] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "script":
            self._in_script = True
            self._current_attrs = {name: (value or "") for name, value in attrs}
            self._current_data = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_script:
            content = "".join(self._current_data)
            self.scripts.append((self._current_attrs, content))
            self._in_script = False
            self._current_attrs = {}
            self._current_data = []

    def handle_data(self, data: str) -> None:
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


def extract_loader_payload(html: str) -> Optional[List[JsonValue]]:
    """Extract the React Flight loader payload if present."""

    for _attrs, text in _extract_scripts(html):
        if not text or "streamController.enqueue" not in text:
            continue
        decoder = json.JSONDecoder()
        start = 0
        while True:
            anchor = text.find("streamController.enqueue(", start)
            if anchor == -1:
                break
            anchor += len("streamController.enqueue(")
            quote_pos = text.find("\"", anchor)
            next_close = text.find(");", anchor)
            if quote_pos != -1 and (next_close == -1 or quote_pos < next_close):
                try:
                    chunk, end_offset = decoder.raw_decode(text, quote_pos)
                except json.JSONDecodeError:
                    start = anchor + 1
                    continue
                start = end_offset
            else:
                end = text.find(");", anchor)
                if end == -1:
                    break
                chunk = text[anchor:end].strip()
                if chunk.startswith("(") and chunk.endswith(")"):
                    chunk = chunk[1:-1].strip()
                start = end + 2
            if isinstance(chunk, str):
                chunk = chunk.strip()
            if isinstance(chunk, str) and chunk.startswith("["):
                try:
                    parsed_chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    parsed_chunk = None
                if isinstance(parsed_chunk, list):
                    return cast(List[JsonValue], parsed_chunk)
    return None


def decode_loader(loader: List[JsonValue]) -> Dict[str, JsonValue]:
    """Decode the flattened loader list into dictionaries and lists."""

    cache: Dict[int, JsonValue] = {}

    def decode_key(raw_key: JsonValue) -> str:
        if isinstance(raw_key, str) and raw_key.startswith("_") and raw_key[1:].isdigit():
            idx = int(raw_key[1:])
            if 0 <= idx < len(loader):
                candidate = loader[idx]
                if isinstance(candidate, str):
                    return candidate
        return str(raw_key)

    def resolve(value: JsonValue) -> JsonValue:
        if type(value) is int:
            if value in cache:
                return cache[value]
            if not (0 <= value < len(loader)):
                return cast(JsonValue, value)
            cache[value] = cast(JsonValue, None)
            resolved_value = resolve(loader[value])
            cache[value] = resolved_value
            return resolved_value
        if isinstance(value, list):
            return cast(JsonValue, [resolve(item) for item in value])
        if isinstance(value, dict):
            return cast(
                JsonValue,
                {decode_key(k): resolve(v) for k, v in value.items()},
            )
        return value

    resolved: Dict[str, JsonValue] = {}
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
    if loader is None:
        raise ValueError("Modern share payload not found")

    decoded = decode_loader(loader)
    loader_data = cast(Mapping[str, Any], decoded.get("loaderData", {}))
    route = cast(Mapping[str, Any], loader_data.get("routes/share.$shareId.($action)", {}))
    server_response = cast(Mapping[str, Any], route.get("serverResponse", {}))
    data = cast(Mapping[str, Any], server_response.get("data", {}))
    share_id_value = route.get("sharedConversationId")
    share_id = share_id_value if isinstance(share_id_value, str) else "shared"
    model = cast(Mapping[str, Any], data.get("model", {}))
    model_slug_value = model.get("slug")
    model_slug = model_slug_value if isinstance(model_slug_value, str) else ""
    title_value = data.get("title")
    title = title_value if isinstance(title_value, str) else ""
    updated_raw = data.get("update_time")
    if isinstance(updated_raw, (int, float)):
        updated_at: Optional[float] = float(updated_raw)
    else:
        updated_at = None
    mapping = cast(Mapping[str, Any], data.get("mapping", {}))
    sequence_field = data.get("linear_conversation", [])
    sequence: List[Mapping[str, Any]] = (
        [entry for entry in sequence_field if isinstance(entry, Mapping)]
        if isinstance(sequence_field, list)
        else []
    )

    replies: List[Reply] = []
    for entry in sequence:
        node_id_raw = entry.get("id") if isinstance(entry, Mapping) else None
        if not isinstance(node_id_raw, str):
            continue
        node = mapping.get(node_id_raw)
        if not node:
            continue
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not message:
            continue
        if not isinstance(message, Mapping):
            continue
        author_info = message.get("author") or {}
        role = author_info.get("role") if isinstance(author_info, Mapping) else None
        if role == "system":
            continue
        content = message.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        message_id = cast(Optional[str], message.get("id"))
        statement, assets = flatten_message_content(message_id, content, message)
        if not statement and not assets:
            continue
        reply_type = ReplyType(role) if role in ReplyType._value2member_map_ else ReplyType.AI
        author = author_name_for_role(role)
        created_raw = message.get("create_time")
        created_at = float(created_raw) if isinstance(created_raw, (int, float)) else None
        replies.append(
            Reply(
                author_name=author,
                type=reply_type,
                statement=statement,
                created_at=created_at,
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
    payload = cast(Dict[str, Any], json.loads(script_content))
    props = cast(Mapping[str, Any], payload.get("props", {}))
    page_props = cast(Mapping[str, Any], props.get("pageProps", {}))
    server_response = cast(Mapping[str, Any], page_props.get("serverResponse", {}))
    data = cast(Mapping[str, Any], server_response.get("data", {}))
    share_id = cast(str, data.get("conversation_id", "shared"))
    model = cast(Mapping[str, Any], data.get("model", {}))
    model_slug_value = model.get("slug")
    model_slug = model_slug_value if isinstance(model_slug_value, str) else ""
    title_value = data.get("title")
    title = title_value if isinstance(title_value, str) else ""
    updated_raw = data.get("update_time")
    if isinstance(updated_raw, (int, float)):
        updated_at: Optional[float] = float(updated_raw)
    else:
        updated_at = None
    author_name_raw = data.get("author_name", "User")
    author_name = author_name_raw if isinstance(author_name_raw, str) else "User"
    sequence = cast(List[Mapping[str, Any]], data.get("linear_conversation", []))

    replies: List[Reply] = []
    for node in sequence:
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        author_info = message.get("author") or {}
        role = author_info.get("role") if isinstance(author_info, Mapping) else None
        if role == "system":
            continue
        content = message.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        message_id = cast(Optional[str], message.get("id"))
        statement, assets = flatten_message_content(message_id, content, message)
        if not statement and not assets:
            continue
        author = author_name if role == "user" else author_name_for_role(role)
        reply_type = ReplyType(role) if role in ReplyType._value2member_map_ else ReplyType.AI
        created_raw = message.get("create_time")
        created_at = float(created_raw) if isinstance(created_raw, (int, float)) else None
        replies.append(
            Reply(
                author_name=author,
                type=reply_type,
                statement=statement,
                created_at=created_at,
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
CITATION_TOKEN_PATTERN = re.compile(r"\s*(?:citeturn|navlist|turn\d+\w*)[^,\s]*,?")


def summarize_tool_payload(data: Mapping[str, Any]) -> Optional[str]:
    lines: List[str] = []

    search_queries = data.get("search_query")
    queries: List[str] = []
    if isinstance(search_queries, list):
        for entry in search_queries:
            if isinstance(entry, Mapping):
                query = entry.get("q")
                if isinstance(query, str):
                    query = query.strip()
                    if query:
                        queries.append(query)
            elif isinstance(entry, str):
                query = entry.strip()
                if query:
                    queries.append(query)
    if queries:
        lines.append("Search tool invoked with queries:")
        lines.extend(f"- {query}" for query in queries)

    additional_items: List[str] = []
    for key, value in data.items():
        if key in {"search_query", "response_length"}:
            continue
        if isinstance(value, (str, int, float)):
            value_str = str(value).strip()
            if value_str:
                additional_items.append(f"{key}: {value_str}")
    if additional_items:
        if not lines:
            lines.append("Tool parameters:")
        lines.extend(f"- {item}" for item in additional_items)

    if lines:
        return "\n".join(lines)
    return None


def strip_private_use(text: str) -> str:
    return PRIVATE_USE_PATTERN.sub("", text)


def strip_citation_tokens(text: str) -> str:
    if not text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        cleaned = CITATION_TOKEN_PATTERN.sub("", line).rstrip()
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def flatten_message_content(
    message_id: Optional[str],
    content: Mapping[str, Any],
    message: Mapping[str, Any],
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
        metadata_raw = message.get("metadata") or {}
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        attachment_lines: List[str] = []

        attachments_field = metadata.get("attachments", [])
        if isinstance(attachments_field, list):
            for attachment_raw in attachments_field:
                if not isinstance(attachment_raw, Mapping):
                    continue
                url_raw = attachment_raw.get("download_url") or attachment_raw.get("file_url")
                url = url_raw if isinstance(url_raw, str) else None
                if not url:
                    continue
                mime = attachment_raw.get("mime_type")
                mime_str = mime if isinstance(mime, str) else None
                name_value = attachment_raw.get("name")
                name = name_value if isinstance(name_value, str) else None
                filename = name or build_asset_filename(message_id, len(assets), mime_str)
                asset_type_raw = (
                    attachment_raw.get("file_type")
                    or attachment_raw.get("type")
                    or "file"
                )
                asset_type = asset_type_raw if isinstance(asset_type_raw, str) else "file"
                downloadable = url.lower().startswith("http")
                description_raw = attachment_raw.get("title") or attachment_raw.get("name")
                description = description_raw if isinstance(description_raw, str) else None
                assets.append(
                    ConversationAsset(
                        asset_type="image" if "image" in asset_type.lower() else "file",
                        url=url,
                        filename=filename,
                        description=description,
                        downloadable=downloadable,
                    )
                )
                attachment_lines.append(render_asset_reference(assets[-1]))
        combined = text.strip()
        if attachment_lines:
            combined = (combined + "\n\n" if combined else "") + "\n".join(attachment_lines)
        combined = strip_citation_tokens(combined)
        return combined, assets

    if content_type == "text":
        parsed_parts: List[str] = []
        parts_field = content.get("parts", [])
        if isinstance(parts_field, list):
            for part in parts_field:
                if not isinstance(part, str):
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
                            fallback = maybe_json.get("content")
                            parsed = fallback if isinstance(fallback, str) else cleaned
                parsed_parts.append(parsed)
        parts = parsed_parts
        return finalize("\n\n".join(part for part in parts if part))

    if content_type == "code":
        language = content.get("language")
        code_text = content.get("text", "")
        lang = language if isinstance(language, str) and language != "unknown" else ""
        text_body = code_text if isinstance(code_text, str) else ""
        body = text_body.rstrip("\n")
        if body:
            try:
                maybe_json = json.loads(body)
            except json.JSONDecodeError:
                maybe_json = None
            if isinstance(maybe_json, Mapping):
                summary = summarize_tool_payload(maybe_json)
                if summary is not None:
                    return finalize(summary)
                cleaned_dict = {
                    key: value
                    for key, value in maybe_json.items()
                    if key != "response_length"
                }
                if not cleaned_dict:
                    return finalize("")
                body = json.dumps(cleaned_dict, indent=2, ensure_ascii=False)
        return finalize(f"```{lang}\n{body}\n```")

    if content_type == "thoughts":
        thoughts: List[str] = []
        thoughts_field = content.get("thoughts", [])
        if isinstance(thoughts_field, list):
            for thought in thoughts_field:
                if not isinstance(thought, Mapping):
                    continue
                summary_raw = thought.get("summary")
                detail_raw = thought.get("content")
                summary = summary_raw if isinstance(summary_raw, str) else None
                detail = detail_raw if isinstance(detail_raw, str) else None
                combined = ": ".join(filter(None, [summary, detail]))
                if combined:
                    thoughts.append(f"_{combined}_")
        return finalize("\n\n".join(thoughts))

    if content_type == "reasoning_recap":
        recap_raw = content.get("content", "")
        recap = recap_raw if isinstance(recap_raw, str) else ""
        return finalize(f"_{recap.strip()}_" if recap else "")

    if content_type == "model_editable_context":
        model_context = content.get("model_set_context", "")
        if isinstance(model_context, str):
            return finalize(model_context.strip())
        return finalize("")

    if content_type == "multimodal_text":
        segments: List[str] = []
        parts_field = content.get("parts", [])
        if isinstance(parts_field, list):
            for part in parts_field:
                if isinstance(part, str):
                    segments.append(strip_private_use(part))
                    continue
                if not isinstance(part, Mapping):
                    continue
                p_type_raw = part.get("content_type") or part.get("type")
                p_type = p_type_raw if isinstance(p_type_raw, str) else None
                if p_type == "text":
                    texts = part.get("text")
                    if isinstance(texts, list):
                        segments.extend(
                            strip_private_use(t) for t in texts if isinstance(t, str)
                        )
                    elif isinstance(texts, str):
                        segments.append(strip_private_use(texts))
                elif p_type in {"image_asset_pointer", "file"}:
                    pointer_raw = part.get("asset_pointer")
                    if not isinstance(pointer_raw, str) or not pointer_raw:
                        continue
                    mime_value = part.get("mime_type")
                    mime = mime_value if isinstance(mime_value, str) else None
                    filename = build_asset_filename(message_id, len(assets), mime)
                    asset_type = "image" if "image" in (p_type or "").lower() else "file"
                    downloadable = pointer_raw.lower().startswith("http")
                    assets.append(
                        ConversationAsset(
                            asset_type=asset_type,
                            url=pointer_raw,
                            filename=filename,
                            downloadable=downloadable,
                        )
                    )
                    segments.append(render_asset_reference(assets[-1]))
        return finalize("\n\n".join(segment.strip() for segment in segments if segment.strip()))

    if content_type == "tool_response":
        output_raw = content.get("output", "")
        output = output_raw if isinstance(output_raw, str) else ""
        return finalize(strip_private_use(output))

    # Attempt a generic text conversion as last resort
    if "parts" in content:
        parts_field = content.get("parts", [])
        if isinstance(parts_field, list):
            parts = [strip_private_use(str(part)) for part in parts_field if part]
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

    def __init__(self, link: str) -> None:
        self._link: str = link
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
