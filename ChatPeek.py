"""Utilities for exporting ChatGPT and Claude shared conversations to Markdown."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
from urllib.parse import urljoin, urlparse

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

# Hosts ChatGPT/OpenAI serve conversation assets from. Asset URLs come from the
# untrusted share page, so downloads are limited to these hosts (or their
# subdomains); anything else is rendered as a placeholder instead of being
# fetched. This is the initial-URL half of the SSRF defense; default_http_get
# also refuses to follow redirects so an allowed host cannot bounce the fetch
# to an internal address.
ALLOWED_ASSET_HOST_SUFFIXES: Tuple[str, ...] = (
    "oaiusercontent.com",
    "oaistatic.com",
    "chatgpt.com",
    "openai.com",
    # DALL-E images are served from this OpenAI-owned Azure storage account.
    # Only this exact account is allowed: matching all of *.blob.core.windows.net
    # would let a share page point downloads at an attacker's own Azure storage.
    # Assets on other (newer) OpenAI storage accounts degrade to a placeholder
    # rather than being fetched — a deliberate security-over-recall tradeoff.
    "oaidalleapiprodscus.blob.core.windows.net",
)

_ALLOWED_ASSET_HOST_DOTTED: Tuple[str, ...] = tuple(
    "." + suffix for suffix in ALLOWED_ASSET_HOST_SUFFIXES
)


def is_allowed_asset_url(url: object) -> bool:
    """Return True if url is an https link to a known ChatGPT/OpenAI asset host."""

    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    # https only: real asset URLs are always https, and requiring it blocks
    # cleartext fetches an attacker could otherwise steer through an allowed host.
    if parsed.scheme != "https":
        return False
    # rstrip('.') normalizes a fully-qualified "host." form to its bare host.
    host = (parsed.hostname or "").lower().rstrip(".")
    return host in ALLOWED_ASSET_HOST_SUFFIXES or host.endswith(_ALLOWED_ASSET_HOST_DOTTED)


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


@dataclass(frozen=True)
class ExportOptions:
    """Controls which internal conversation details are included in exports."""

    include_reasoning: bool = False
    include_tool_output: bool = False
    include_model_context: bool = False


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


def _strip_line_end(line: str) -> str:
    """Trim trailing whitespace, keeping a Markdown hard line break.

    Two or more trailing spaces after text are Markdown's hard break, so they
    are normalised to exactly two rather than dropped.
    """

    stripped = line.rstrip()
    if stripped and line.endswith("  ") and line[len(stripped):].strip(" ") == "":
        return stripped + "  "
    return stripped


@dataclass
class Chat:
    """Structured representation of a shared ChatGPT or Claude conversation."""

    share_id: str
    ai_model: str
    title: str
    updated_at: Optional[float]
    replies: List[Reply]
    source: str = "ChatGPT"

    def to_markdown(self) -> str:
        """Render the conversation as a Markdown string."""

        header_lines = [f"# {self.title or self.source + ' conversation'}"]
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

        return "\n".join(_strip_line_end(line) for line in "\n".join(header_lines).split("\n")).rstrip() + "\n"

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
            resolved_parents = {True: images_dir.resolve(), False: files_dir.resolve()}
            fetch = http_get or default_http_get
            for reply in self.replies:
                for asset in reply.assets:
                    if not asset.downloadable or not is_allowed_asset_url(asset.url):
                        continue
                    is_image = asset.asset_type == "image"
                    target_dir = images_dir if is_image else files_dir
                    # Asset filenames originate from untrusted share-page JSON, so
                    # require a bare path component before building any path.
                    # PureWindowsPath applies the stricter (Windows) rules on every
                    # platform, which is what makes this check platform-independent:
                    # on POSIX, "C:\evil.txt" and "\\host\share\evil.txt" are legal
                    # single filenames, so the containment check below passes them
                    # and the export ends up holding a file no Windows machine can
                    # open and no Markdown link resolves to.
                    if PureWindowsPath(asset.filename).name != asset.filename:
                        continue
                    target_path = target_dir / asset.filename
                    try:
                        # Second layer: even a bare component must resolve to a
                        # direct child of the target folder, so a symlink or a name
                        # that slipped through above still cannot escape.
                        if target_path.resolve().parent != resolved_parents[is_image]:
                            continue
                        target_dir.mkdir(exist_ok=True)
                        if target_path.exists():
                            continue
                        resp = fetch(asset.url)
                        resp.raise_for_status()
                        # A non-allowed redirect returns a non-200 status; skip it
                        # rather than write the redirect body as the asset.
                        if getattr(resp, "status_code", 200) != 200:
                            continue
                        target_path.write_bytes(resp.content)
                    except (requests.RequestException, OSError):
                        # One malformed URL or unwritable filename must not abort
                        # the export of an otherwise-valid conversation.
                        continue

        return markdown_path


class ShareAccessError(RuntimeError):
    """Raised when a share URL cannot be fetched due to access restrictions."""


_MAX_ASSET_REDIRECTS = 5

# Retry policy for the share fetch. Only failures that a later attempt could
# plausibly resolve are retried -- see fetch_share_page for why everything else
# is treated as final.
DEFAULT_FETCH_ATTEMPTS: int = 3
_RETRY_BACKOFF_SECONDS: float = 5.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Transport faults worth another attempt. ChunkedEncodingError and
# ContentDecodingError surface from requests.get itself rather than from a
# status code: the default stream=False reads the body during send(), so a
# connection dropped mid-body or a truncated gzip is raised there.
_RETRYABLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
    # Only reachable if a caller mounts an adapter with max_retries set, since
    # the requests default is 0. When it does fire the underlying cause was
    # transient, so it belongs here -- note the attempts below then compose with
    # that adapter's own retry budget rather than replacing it.
    requests.exceptions.RetryError,
)

# TooManyRedirects is deliberately absent: a redirect loop is a property of the
# URL, so it is final in the same way a 404 is.

# Subclasses of ConnectionError whose cause is configuration rather than
# congestion: a rejected certificate or an unreachable proxy answers the same
# way every time, so retrying only delays an error the caller has to fix.
_FINAL_CONNECTION_EXCEPTIONS = (
    requests.exceptions.SSLError,
    requests.exceptions.ProxyError,
)

# A throttled response names its own delay. Honour it in preference to the
# backoff -- a fixed 5s wait against a limiter asking for 60s just re-trips it --
# but cap it so a hostile or absurd Retry-After cannot park the CLI.
_MAX_RETRY_AFTER_SECONDS: float = 60.0


def _positive_int(value: str) -> int:
    """argparse type for a count that must be at least 1."""

    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {parsed}")
    return parsed


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Seconds to wait before the next attempt.

    Prefers a usable Retry-After header, falling back to the linear backoff when
    the header is absent, non-numeric (the RFC also permits an HTTP-date form),
    or non-positive.
    """

    backoff = _RETRY_BACKOFF_SECONDS * attempt
    if response is None:
        return backoff
    try:
        requested = float(response.headers["Retry-After"])
    except (KeyError, TypeError, ValueError):
        return backoff
    return min(requested, _MAX_RETRY_AFTER_SECONDS) if requested > 0 else backoff


def default_http_get(url: str) -> requests.Response:
    """Fetch an asset, following redirects only to other allowed asset hosts.

    Redirects are resolved manually (allow_redirects=False) so a hop to an
    off-allowlist host — e.g. an internal address reached via an open redirect
    on an allowed host — is refused *before* the request is made. This keeps the
    SSRF guard intact while still supporting allowed hosts (such as ChatGPT
    backend file endpoints) that legitimately redirect to a signed blob URL.
    """

    current = url
    response = requests.get(current, headers=DEFAULT_HEADERS, timeout=30, allow_redirects=False)
    for _ in range(_MAX_ASSET_REDIRECTS):
        if not response.is_redirect:
            return response
        location = response.headers.get("Location")
        if not location:
            return response
        current = urljoin(current, location)
        if not is_allowed_asset_url(current):
            # Refuse to follow a redirect off the allowlist; the caller sees the
            # non-200 status and skips the asset rather than fetching internally.
            return response
        response = requests.get(current, headers=DEFAULT_HEADERS, timeout=30, allow_redirects=False)
    return response


def fetch_share_page(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    attempts: int = DEFAULT_FETCH_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fetch the shared conversation HTML using private-window style headers.

    Only *transient* failures are retried: the transport faults in
    _RETRYABLE_EXCEPTIONS and the statuses in _RETRYABLE_STATUS_CODES. A
    deterministic answer -- a private conversation, a deleted share, any other
    4xx, a rejected certificate -- is raised on the first attempt, because a
    retry would return the same result and only add load to a server that
    already answered the question.

    Args:
        url: The https://chatgpt.com/share/... link to fetch.
        headers: Extra headers merged over the browser-style defaults.
        timeout: Per-attempt timeout in seconds.
        attempts: Maximum attempts before giving up; 1 disables retrying.
        sleep: Injector for the backoff delay (facilitates testing).

    Returns:
        The HTML body of the share page.

    Raises:
        ShareAccessError: The link points at a private conversation.
        requests.RequestException: The final attempt failed.
    """

    merged_headers = {**DEFAULT_HEADERS, "Referer": "https://chatgpt.com/"}
    if headers:
        merged_headers.update(headers)

    total = max(attempts, 1)
    for attempt in range(1, total + 1):
        response: Optional[requests.Response] = None
        try:
            response = requests.get(url, headers=merged_headers, timeout=timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Read the status off the exception rather than the loop-local
            # response: the two disagree if requests.get itself ever raises, and
            # a stale status from a previous attempt would misroute this one.
            failed = exc.response
            status = failed.status_code if failed is not None else None
            parsed = urlparse(url)
            path = parsed.path or ""
            if status == 403 and parsed.netloc.endswith("chatgpt.com") and path.startswith("/c/"):
                raise ShareAccessError(
                    "The provided link appears to be a private conversation. "
                    "Open it while logged in and copy the public https://chatgpt.com/share/... link instead."
                ) from exc
            if attempt == total or status not in _RETRYABLE_STATUS_CODES:
                raise
            response = failed
        except _FINAL_CONNECTION_EXCEPTIONS:
            raise
        except _RETRYABLE_EXCEPTIONS:
            if attempt == total:
                raise
            response = None
        else:
            return response.text
        sleep(_retry_delay(response, attempt))

    # Unreachable: the final attempt either returns or re-raises above. Kept so
    # the function has a return path the type checker can prove. Deliberately
    # not a ShareAccessError, which check_fixture_links maps to "stale link".
    raise RuntimeError(f"Could not fetch {url} after {total} attempts.")


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


def parse_modern_share(html: str, options: Optional[ExportOptions] = None) -> Chat:
    export_options = options or ExportOptions()
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

    namer = _AssetNamer()
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
        statement, assets = flatten_message_content(message_id, content, message, export_options, namer)
        if not statement and not assets:
            continue
        if role == "tool" and is_redacted_tool_output(statement):
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


def parse_post_share(html: str, options: Optional[ExportOptions] = None) -> Chat:
    export_options = options or ExportOptions()
    loader = extract_loader_payload(html)
    if loader is None:
        raise ValueError("Post share payload not found")

    decoded = decode_loader(loader)
    loader_data = decoded.get("loaderData")
    if not isinstance(loader_data, Mapping):
        raise ValueError("Post share route not found")
    route = loader_data.get("routes/s.$postId")
    if not isinstance(route, Mapping):
        raise ValueError("Post share route not found")
    post_with_profile_raw = route.get("postWithProfile")
    post_with_profile: Mapping[str, Any] = (
        post_with_profile_raw if isinstance(post_with_profile_raw, Mapping) else {}
    )
    post_raw = post_with_profile.get("post")
    post: Mapping[str, Any] = post_raw if isinstance(post_raw, Mapping) else {}
    share_id_value = post.get("id")
    share_id = share_id_value if isinstance(share_id_value, str) else "shared"
    title_value = post.get("text")
    title = title_value if isinstance(title_value, str) else ""
    posted_at = post.get("posted_at")
    updated_at = float(posted_at) if isinstance(posted_at, (int, float)) else None

    messages: List[Mapping[str, Any]] = []
    attachments = post.get("attachments", [])
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                continue
            if attachment.get("kind") != "message_slice":
                continue
            raw_messages = attachment.get("messages", [])
            if isinstance(raw_messages, list):
                messages.extend(
                    message for message in raw_messages if isinstance(message, Mapping)
                )

    namer = _AssetNamer()
    replies: List[Reply] = []

    def append_message(message: Mapping[str, Any]) -> None:
        author_info = message.get("author") or {}
        role = author_info.get("role") if isinstance(author_info, Mapping) else None
        if role == "system":
            return
        content = message.get("content") or {}
        if not isinstance(content, Mapping):
            return
        message_id = cast(Optional[str], message.get("id"))
        statement, assets = flatten_message_content(message_id, content, message, export_options, namer)
        if not statement and not assets:
            return
        if role == "tool" and is_redacted_tool_output(statement):
            return
        reply_type = ReplyType(role) if role in ReplyType._value2member_map_ else ReplyType.AI
        created_raw = message.get("create_time")
        created_at = float(created_raw) if isinstance(created_raw, (int, float)) else None
        replies.append(
            Reply(
                author_name=author_name_for_role(role),
                type=reply_type,
                statement=statement,
                created_at=created_at,
                assets=assets,
            )
        )

    for message in messages:
        append_message(message)
        metadata = message.get("metadata") or {}
        metadata_map = metadata if isinstance(metadata, Mapping) else {}
        chatgpt_sdk = metadata_map.get("chatgpt_sdk") or {}
        chatgpt_sdk_map = chatgpt_sdk if isinstance(chatgpt_sdk, Mapping) else {}
        widget_state_raw = chatgpt_sdk_map.get("widget_state")
        if not isinstance(widget_state_raw, str):
            continue
        try:
            widget_state = json.loads(widget_state_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(widget_state, Mapping):
            continue
        report_message = widget_state.get("report_message")
        if isinstance(report_message, Mapping):
            append_message(report_message)

    return Chat(share_id=share_id, ai_model="", title=title, updated_at=updated_at, replies=replies)


def parse_legacy_share(html: str, options: Optional[ExportOptions] = None) -> Chat:
    export_options = options or ExportOptions()
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

    namer = _AssetNamer()
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
        statement, assets = flatten_message_content(message_id, content, message, export_options, namer)
        if not statement and not assets:
            continue
        if role == "tool" and is_redacted_tool_output(statement):
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


def parse_share_html(html: str, options: Optional[ExportOptions] = None) -> Chat:
    loader = extract_loader_payload(html)
    if loader is not None:
        decoded = decode_loader(loader)
        loader_data = decoded.get("loaderData")
        if isinstance(loader_data, Mapping) and "routes/s.$postId" in loader_data:
            return parse_post_share(html, options)

    try:
        return parse_modern_share(html, options)
    except (ValueError, KeyError):
        try:
            return parse_post_share(html, options)
        except (ValueError, KeyError):
            return parse_legacy_share(html, options)


# --- Claude (claude.ai) shares -------------------------------------------------
#
# A claude.ai/share/<uuid> page is a client-rendered shell with no conversation
# in it: the browser fetches the snapshot from a public JSON endpoint instead.
# ChatPeek requests that endpoint directly, the same one the page itself calls.

CLAUDE_SHARE_PATTERN = re.compile(
    r"^/share/([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})/?$"
)
CLAUDE_SNAPSHOT_URL = (
    "https://claude.ai/api/chat_snapshots/{share_id}"
    "?rendering_mode=messages&render_all_tools=true"
)

# Artifact MIME types mapped to the fence language used for their source.
_CLAUDE_ARTIFACT_LANGUAGES: Dict[str, str] = {
    "text/markdown": "markdown",
    "text/html": "html",
    "image/svg+xml": "svg",
    "application/vnd.ant.react": "jsx",
    "application/vnd.ant.mermaid": "mermaid",
}


def claude_share_id(url: str) -> Optional[str]:
    """Return the share UUID if url is a https://claude.ai/share/<uuid> link."""

    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in ("http", "https") or host not in ("claude.ai", "www.claude.ai"):
        return None
    match = CLAUDE_SHARE_PATTERN.match(parsed.path or "")
    return match.group(1).lower() if match else None


def _code_fence(body: str, language: str = "") -> str:
    """Fence body in backticks longer than any run it contains."""

    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{body.rstrip()}\n{fence}"


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.strip().splitlines())


def _parse_iso_timestamp(value: object) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        # fromisoformat accepts a trailing "Z" from Python 3.11 on; normalise it
        # anyway so the parse does not depend on the interpreter version.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _render_claude_artifact(tool_input: Mapping[str, Any]) -> str:
    """Render an artifacts tool call as the artifact's source code."""

    title_raw = tool_input.get("title")
    title = title_raw if isinstance(title_raw, str) and title_raw else "Untitled artifact"
    command = tool_input.get("command")
    if command == "update":
        old_raw = tool_input.get("old_str")
        new_raw = tool_input.get("new_str")
        old = old_raw if isinstance(old_raw, str) else ""
        new = new_raw if isinstance(new_raw, str) else ""
        return "\n\n".join(
            [f"**Artifact updated: {title}**", "Replaced:", _code_fence(old), "With:", _code_fence(new)]
        )
    content_raw = tool_input.get("content")
    if not isinstance(content_raw, str) or not content_raw.strip():
        return ""
    language_raw = tool_input.get("language")
    artifact_type = tool_input.get("type")
    if isinstance(language_raw, str) and language_raw:
        language = language_raw
    elif isinstance(artifact_type, str):
        language = _CLAUDE_ARTIFACT_LANGUAGES.get(artifact_type, "")
    else:
        language = ""
    return f"**Artifact: {title}**\n\n{_code_fence(content_raw, language)}"


def _render_claude_tool_result(item: Mapping[str, Any]) -> str:
    name_raw = item.get("name")
    name = name_raw if isinstance(name_raw, str) and name_raw else "tool"
    texts: List[str] = []
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            else:
                title = part.get("title")
                url = part.get("url")
                if isinstance(title, str) and isinstance(url, str):
                    texts.append(f"{title} - {url}")
    elif isinstance(content, str) and content.strip():
        texts.append(content.strip())
    if not texts:
        return ""
    label = "Tool error" if item.get("is_error") else "Tool result"
    return f"*{label} ({name}):*\n\n{_code_fence(chr(10).join(texts), 'text')}"


def _render_claude_content(item: Mapping[str, Any], options: ExportOptions) -> str:
    content_type = item.get("type")
    if content_type == "text":
        text = item.get("text")
        return text.strip() if isinstance(text, str) else ""
    if content_type == "thinking":
        if not options.include_reasoning:
            return ""
        thinking = item.get("thinking")
        if not isinstance(thinking, str) or not thinking.strip():
            return ""
        return _blockquote(f"**Thinking**\n\n{thinking.strip()}")
    if content_type == "voice_note":
        text = item.get("text")
        return text.strip() if isinstance(text, str) else ""
    if content_type == "tool_use":
        tool_input_raw = item.get("input")
        tool_input: Mapping[str, Any] = tool_input_raw if isinstance(tool_input_raw, Mapping) else {}
        # Artifacts are the answer itself rather than tool plumbing, so they are
        # always exported; every other tool call is opt-in like on ChatGPT.
        if item.get("name") == "artifacts":
            return _render_claude_artifact(tool_input)
        if not options.include_tool_output:
            return ""
        name_raw = item.get("name")
        name = name_raw if isinstance(name_raw, str) and name_raw else "tool"
        body = json.dumps(tool_input, indent=2, ensure_ascii=False)
        return f"*Tool call: {name}*\n\n{_code_fence(body, 'json')}"
    if content_type == "tool_result":
        if not options.include_tool_output or item.get("name") == "artifacts":
            return ""
        return _render_claude_tool_result(item)
    return ""


def _render_claude_files(message: Mapping[str, Any]) -> List[str]:
    """Describe pasted text, uploaded documents and images on a message.

    Uploaded files are only reachable with the sharer's session cookie, so
    images are noted by name rather than downloaded; documents whose text
    claude.ai extracted are inlined in a collapsible block.
    """

    blocks: List[str] = []
    attachments = message.get("attachments")
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                continue
            name_raw = attachment.get("file_name")
            name = name_raw if isinstance(name_raw, str) and name_raw else "Pasted content"
            extracted = attachment.get("extracted_content")
            if isinstance(extracted, str) and extracted.strip():
                blocks.append(
                    f"<details>\n<summary>Attachment: {name}</summary>\n\n"
                    f"{_code_fence(extracted, 'text')}\n\n</details>"
                )
            else:
                blocks.append(f"*Attachment '{name}' not included in export.*")

    seen: Set[str] = set()
    for key in ("files_v2", "files"):
        files = message.get(key)
        if not isinstance(files, list):
            continue
        for entry in files:
            if not isinstance(entry, Mapping):
                continue
            identity = entry.get("file_uuid") or entry.get("file_name")
            if isinstance(identity, str):
                if identity in seen:
                    continue
                seen.add(identity)
            name_raw = entry.get("file_name")
            name = name_raw if isinstance(name_raw, str) and name_raw else "file"
            kind = "Image" if entry.get("file_kind") == "image" else "File"
            blocks.append(f"*{kind} '{name}' not included in export.*")
    return blocks


def parse_claude_snapshot(body: str, options: Optional[ExportOptions] = None) -> Chat:
    """Build a Chat from a claude.ai chat_snapshots JSON response."""

    export_options = options or ExportOptions()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Claude share response is not JSON") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("chat_messages"), list):
        raise ValueError("Claude share response has no chat_messages")

    share_id_raw = data.get("uuid")
    share_id = share_id_raw if isinstance(share_id_raw, str) and share_id_raw else "shared"
    title = ""
    for key in ("snapshot_name", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            title = value.strip()
            break
    model_raw = data.get("model")
    model = model_raw if isinstance(model_raw, str) else ""
    updated_at = _parse_iso_timestamp(data.get("updated_at")) or _parse_iso_timestamp(
        data.get("created_at")
    )

    messages = [m for m in data["chat_messages"] if isinstance(m, Mapping)]
    # index is the position in the shared branch; fall back to list order.
    def message_index(message: Mapping[str, Any]) -> int:
        index = message.get("index")
        return index if isinstance(index, int) else 0

    messages.sort(key=message_index)

    replies: List[Reply] = []
    for message in messages:
        sender = message.get("sender")
        content = message.get("content")
        parts: List[str] = []
        if isinstance(content, list) and content:
            for item in content:
                if isinstance(item, Mapping):
                    rendered = _render_claude_content(item, export_options)
                    if rendered:
                        parts.append(rendered)
        else:
            # Older snapshots carry only the flat text field.
            text = message.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        parts.extend(_render_claude_files(message))
        statement = "\n\n".join(parts)
        if not statement:
            continue
        is_human = sender == "human"
        replies.append(
            Reply(
                author_name="User" if is_human else "Claude",
                type=ReplyType.HUMAN if is_human else ReplyType.AI,
                statement=statement,
                created_at=_parse_iso_timestamp(message.get("created_at")),
            )
        )

    return Chat(
        share_id=share_id,
        ai_model=model,
        title=title,
        updated_at=updated_at,
        replies=replies,
        source="Claude",
    )


def fetch_claude_snapshot(
    share_url: str,
    attempts: int = DEFAULT_FETCH_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fetch the JSON snapshot behind a claude.ai share link.

    Raises:
        ValueError: share_url is not a claude.ai share link.
        ShareAccessError: claude.ai refused the request (unshared link or a
            bot challenge).
        requests.RequestException: Any other failure after retries.
    """

    share_id = claude_share_id(share_url)
    if share_id is None:
        raise ValueError(f"Not a claude.ai share link: {share_url}")
    headers = {
        "Accept": "application/json",
        "Referer": f"https://claude.ai/share/{share_id}",
    }
    try:
        return fetch_share_page(
            CLAUDE_SNAPSHOT_URL.format(share_id=share_id),
            headers=headers,
            attempts=attempts,
            sleep=sleep,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        headers = getattr(exc.response, "headers", None) or {}
        if status == 403 and headers.get("cf-mitigated") == "challenge":
            raise ShareAccessError(
                "claude.ai's bot protection challenged the request (HTTP 403). This is "
                "common from cloud servers and VPNs; retry from a home connection."
            ) from exc
        if status == 403:
            raise ShareAccessError(
                "claude.ai refused the request (HTTP 403). Either the link is no longer "
                "shared, or claude.ai's bot protection challenged it, which is common from "
                "cloud servers and VPNs. Check the link opens in a private browser window, "
                "then retry from a home connection."
            ) from exc
        if status == 404:
            raise ShareAccessError(
                "claude.ai has no shared conversation at this link (HTTP 404). "
                "It may have been unshared or deleted."
            ) from exc
        raise


def load_share(
    link: str,
    options: Optional[ExportOptions] = None,
    attempts: int = DEFAULT_FETCH_ATTEMPTS,
) -> Chat:
    """Fetch and parse a ChatGPT or Claude share link."""

    if claude_share_id(link) is not None:
        return parse_claude_snapshot(fetch_claude_snapshot(link, attempts=attempts), options)
    return parse_share_html(fetch_share_page(link, attempts=attempts), options)


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


def message_recipient(message: Mapping[str, Any]) -> Optional[str]:
    recipient = message.get("recipient")
    if isinstance(recipient, str) and recipient:
        return recipient
    return None


def is_tool_addressed(message: Mapping[str, Any]) -> bool:
    """True when the message is addressed to a tool rather than the user.

    ChatGPT marks user-visible messages with recipient "all"; tool invocations
    carry the tool name (e.g. "web", "web.run", "python") instead.
    """
    recipient = message_recipient(message)
    return recipient is not None and recipient != "all"


def should_include_content(content_type: Optional[str], options: ExportOptions) -> bool:
    if content_type in {"thoughts", "reasoning_recap"}:
        return options.include_reasoning
    if content_type == "tool_response":
        return options.include_tool_output
    if content_type == "model_editable_context":
        return options.include_model_context
    return True


def is_redacted_tool_output(text: str) -> bool:
    return text.strip().lower() == "the output of this plugin was redacted."


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
    options: Optional[ExportOptions] = None,
    namer: Optional["_AssetNamer"] = None,
) -> Tuple[str, List[ConversationAsset]]:
    content_type = content.get("content_type")
    export_options = options or ExportOptions()
    if not should_include_content(content_type, export_options):
        return "", []
    if is_tool_addressed(message) and not export_options.include_tool_output:
        return "", []
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
                filename = sanitize_asset_filename(name, message_id, len(assets), mime_str)
                if namer is not None:
                    filename = namer.assign(url, filename)
                asset_type_raw = (
                    attachment_raw.get("file_type")
                    or attachment_raw.get("type")
                    or "file"
                )
                asset_type = asset_type_raw if isinstance(asset_type_raw, str) else "file"
                downloadable = is_allowed_asset_url(url)
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
        if body and message_recipient(message) != "all":
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
                    if namer is not None:
                        filename = namer.assign(pointer_raw, filename)
                    asset_type = "image" if "image" in (p_type or "").lower() else "file"
                    downloadable = is_allowed_asset_url(pointer_raw)
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


_FILENAME_UNSAFE_PATTERN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Names that resolve to a device rather than a file on Windows.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Cap the on-disk name well under the ~255-byte NAME_MAX / MAX_PATH limits so an
# over-long attacker-supplied name cannot make write_bytes raise OSError. The cap
# is in bytes (not characters) because NAME_MAX is a byte limit, so a multibyte
# name (e.g. CJK) is bounded by its UTF-8 length.
_MAX_FILENAME_BYTES = 200


def _truncate_to_bytes(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    # errors="ignore" drops a trailing partial multibyte sequence.
    return encoded[:budget].decode("utf-8", "ignore")


def _scrub_filename_component(text: str) -> str:
    """Replace filesystem-unsafe characters and trim trailing dots/spaces.

    Windows silently strips trailing dots and spaces when creating a file, so
    they must be removed here to keep the on-disk name matching the Markdown
    link (and to keep the exists() dedupe from aliasing distinct names).
    """

    return _FILENAME_UNSAFE_PATTERN.sub("_", text).strip().rstrip(" .")


def _truncate_filename(name: str) -> str:
    """Bound a filename's UTF-8 byte length, preserving a short extension."""

    if len(name.encode("utf-8")) <= _MAX_FILENAME_BYTES:
        return name
    stem, dot, ext = name.rpartition(".")
    ext_bytes = len(ext.encode("utf-8"))
    if dot and 0 < ext_bytes <= 16:
        return _truncate_to_bytes(stem, _MAX_FILENAME_BYTES - ext_bytes - 1) + "." + ext
    return _truncate_to_bytes(name, _MAX_FILENAME_BYTES)


class _AssetNamer:
    """Allocates conversation-unique on-disk filenames for assets.

    Assets that share a URL reuse one filename, so the download dedupe writes the
    file once and every Markdown link resolves to it. Assets with distinct URLs
    that would otherwise collide on the same basename (e.g. two attachments both
    named "report.csv", or names differing only by a stripped directory prefix)
    get a numeric suffix, so no link ever silently resolves to another asset's
    bytes.
    """

    def __init__(self) -> None:
        self._by_url: Dict[str, str] = {}
        self._used: Set[str] = set()

    def assign(self, url: str, desired: str) -> str:
        existing = self._by_url.get(url)
        if existing is not None:
            return existing
        name = desired
        if name in self._used:
            stem, dot, ext = desired.rpartition(".")
            base = stem if dot else desired
            suffix = dot + ext if dot else ""
            counter = 1
            name = f"{base}-{counter}{suffix}"
            while name in self._used:
                counter += 1
                name = f"{base}-{counter}{suffix}"
        self._used.add(name)
        self._by_url[url] = name
        return name


def build_asset_filename(message_id: Optional[str], index: int, mime_type: Optional[str]) -> str:
    base = _scrub_filename_component((message_id or "asset").split("-")[0]) or "asset"
    extension = ""
    if mime_type and "/" in mime_type:
        # Drop any MIME parameters (e.g. "image/svg+xml; charset=utf-8").
        extension = mime_type.split(";")[0].split("/")[-1]
    extension = _scrub_filename_component(extension) or "bin"
    return f"{base}-{index}.{extension}"


def sanitize_asset_filename(
    name: Optional[str],
    message_id: Optional[str],
    index: int,
    mime_type: Optional[str],
) -> str:
    """Reduce an attachment name from the share page to a safe basename.

    Names come from untrusted JSON, so path separators, traversal sequences,
    drive prefixes, control characters, trailing dots/spaces, over-long names,
    and Windows device names must not survive into the on-disk filename.
    """

    if name:
        # PureWindowsPath treats both / and \ as separators and drops any
        # drive/UNC prefix, so "../../x", "C:x", and "\\host\x" all reduce to the
        # final component regardless of the platform this code runs on.
        candidate = _truncate_filename(_scrub_filename_component(PureWindowsPath(name).name))
        if candidate:
            # Prefix reserved device names (CON, NUL, ...) rather than discard the
            # name: "_aux.pdf" keeps the user's filename while staying writable.
            if candidate.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
                candidate = "_" + candidate
            return candidate
    return build_asset_filename(message_id, index, mime_type)


def slugify_title(title: str, share_id: str) -> str:
    slug_base = re.sub(r"[^a-z0-9]+", "-", (title or "chat").lower()).strip("-") or "chat"
    slug_base = slug_base[:60].rstrip("-")
    return f"{slug_base}-{share_id[:8]}"


class ChatPeek:
    """High-level facade for downloading and exporting shared conversations."""

    def __init__(
        self,
        link: str,
        options: Optional[ExportOptions] = None,
        attempts: int = DEFAULT_FETCH_ATTEMPTS,
    ) -> None:
        self._link: str = link
        self._options = options or ExportOptions()
        self._chat = load_share(link, self._options, attempts=attempts)

    @property
    def chat(self) -> Chat:
        return self._chat


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export a ChatGPT or Claude share link to Markdown")
    parser.add_argument("share_url", help="The https://chatgpt.com/share/... or https://claude.ai/share/... link to export")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_ROOT,
        help="Destination directory for the exported conversation",
    )
    parser.add_argument(
        "--attempts",
        type=_positive_int,
        default=DEFAULT_FETCH_ATTEMPTS,
        help=(
            "Attempts for the share fetch; connection errors, timeouts and "
            "429/5xx responses are retried with a linear backoff (1 disables retrying)"
        ),
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Do not download linked assets (images, attachments)",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include reasoning traces such as thoughts and reasoning recaps",
    )
    parser.add_argument(
        "--include-tool-output",
        action="store_true",
        help="Include tool outputs and tool payload summaries",
    )
    parser.add_argument(
        "--include-model-context",
        action="store_true",
        help="Include model editable context blocks",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    export_options = ExportOptions(
        include_reasoning=args.include_reasoning,
        include_tool_output=args.include_tool_output,
        include_model_context=args.include_model_context,
    )
    try:
        chat = load_share(args.share_url, export_options, attempts=args.attempts)
    except ShareAccessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    markdown_path = chat.save_markdown(args.output, download_assets=not args.skip_assets)
    print(markdown_path)


if __name__ == "__main__":
    main()
