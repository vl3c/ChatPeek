import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, cast
from unittest import mock

import requests

from ChatPeek import (
    Chat,
    ChatPeek,
    JsonValue,
    ShareAccessError,
    decode_loader,
    extract_loader_payload,
    fetch_share_page,
    flatten_message_content,
    parse_modern_share,
    parse_share_html,
    slugify_title,
    strip_private_use,
    strip_citation_tokens,
)


FIXTURES: Path = Path(__file__).resolve().parent / "fixtures"
SHARE_FIXTURE: Path = FIXTURES / "690781ed-75f0-8006-9d6e-d9229bd932f2.html"


class ChatPeekModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html: str = SHARE_FIXTURE.read_text(encoding="utf-8")

    def test_extract_loader_payload(self) -> None:
        payload = extract_loader_payload(self.html)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIsInstance(payload, list)
        self.assertGreater(len(payload), 1000)

    def test_decode_loader_contains_share_id(self) -> None:
        loader = extract_loader_payload(self.html)
        self.assertIsNotNone(loader)
        assert loader is not None
        decoded = decode_loader(loader)
        loader_data = cast(Dict[str, Any], decoded["loaderData"])
        route = cast(Dict[str, Any], loader_data["routes/share.$shareId.($action)"])
        self.assertEqual(route["sharedConversationId"], "690781ed-75f0-8006-9d6e-d9229bd932f2")

    def test_parse_modern_share_returns_chat(self) -> None:
        chat = parse_modern_share(self.html)
        self.assertIsInstance(chat, Chat)
        self.assertEqual(chat.share_id, "690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertIn("Gigawatt", chat.title)
        self.assertGreater(len(chat.replies), 10)

    def test_flatten_message_content_formats_text(self) -> None:
        message: Dict[str, Any] = {
            "id": "abc",
            "content": {"content_type": "text", "parts": ["Hello", "World"]},
            "metadata": {},
        }
        text, assets = flatten_message_content("abc", message["content"], message)
        self.assertEqual(text, "Hello\n\nWorld")
        self.assertEqual(assets, [])

    def test_flatten_message_content_with_code(self) -> None:
        message: Dict[str, Any] = {
            "id": "code",
            "content": {"content_type": "code", "language": "python", "text": "print('hi')\n"},
            "metadata": {},
        }
        text, _ = flatten_message_content("code", message["content"], message)
        self.assertIn("```python", text)
        self.assertIn("print('hi')", text)

    def test_flatten_message_content_adds_attachments(self) -> None:
        message: Dict[str, Any] = {
            "id": "att",
            "content": {"content_type": "text", "parts": ["See file"]},
            "metadata": {
                "attachments": [
                    {
                        "download_url": "https://example.com/file.txt",
                        "name": "file.txt",
                        "mime_type": "text/plain",
                    }
                ]
            },
        }
        text, assets = flatten_message_content("att", message["content"], message)
        self.assertIn("[file.txt](attachments/file.txt)", text)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].filename, "file.txt")

    def test_flatten_message_content_parses_structured_json(self) -> None:
        message: Dict[str, Any] = {
            "id": "json",
            "content": {
                "content_type": "text",
                "parts": [
                    "{\n  \"task_violates_safety_guidelines\": false,\n  \"response\": \"Only include this text\",\n  \"prompt\": \"Ignore this\"\n}"
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content("json", message["content"], message)
        self.assertEqual(text, "Only include this text")
        self.assertEqual(assets, [])

    def test_flatten_message_content_multimodal_file_pointer(self) -> None:
        message: Dict[str, Any] = {
            "id": "msg-1234",
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "file",
                        "asset_pointer": "https://example.com/asset.bin",
                        "mime_type": "application/pdf",
                    }
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content("msg-1234", message["content"], message)
        self.assertIn("[msg-0.pdf](attachments/msg-0.pdf)", text)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].asset_type, "file")

    def test_flatten_message_content_non_downloadable_pointer_adds_note(self) -> None:
        message: Dict[str, Any] = {
            "id": "msg-asset",
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": "sediment://file_123",
                    }
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content("msg-asset", message["content"], message)
        self.assertIn("not included in export", text)
        self.assertEqual(len(assets), 1)
        self.assertFalse(assets[0].downloadable)

    def test_extract_loader_payload_with_semicolons_in_content(self) -> None:
        inner = json.dumps(json.dumps(["data with ); inside", 1, {"key": "val ; ) more"}]))
        html = (
            "<html><script>"
            f'streamController.enqueue({inner});'
            "</script></html>"
        )
        payload = extract_loader_payload(html)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIsInstance(payload, list)
        self.assertEqual(payload[0], "data with ); inside")

    def test_slugify_title_includes_share_suffix(self) -> None:
        slug = slugify_title("Gigawatt Data Centers", "690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertTrue(slug.startswith("gigawatt-data-centers"))
        self.assertTrue(slug.endswith("690781ed"))

    def test_parse_share_html_falls_back_to_legacy(self) -> None:
        legacy_data: Dict[str, Any] = {
            "props": {
                "pageProps": {
                    "serverResponse": {
                        "data": {
                            "conversation_id": "legacy",
                            "title": "Legacy Conversation",
                            "update_time": 1700000000,
                            "model": {"slug": "gpt-3.5"},
                            "author_name": "Tester",
                            "linear_conversation": [
                                {
                                    "message": {
                                        "id": "a",
                                        "author": {"role": "user"},
                                        "content": {"content_type": "text", "parts": ["Hello"]},
                                    }
                                },
                                {
                                    "message": {
                                        "id": "b",
                                        "author": {"role": "assistant"},
                                        "content": {"content_type": "text", "parts": ["Hi there"]},
                                    }
                                },
                            ],
                        }
                    }
                }
            }
        }
        html = f"<html><script id='__NEXT_DATA__'>{json.dumps(legacy_data)}</script></html>"
        chat = parse_share_html(html)
        self.assertEqual(chat.title, "Legacy Conversation")
        self.assertEqual(len(chat.replies), 2)

    @mock.patch("requests.get")
    def test_fetch_share_page_sets_headers(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response
        fetch_share_page("https://chatgpt.com/share/abc")
        args, kwargs = mock_get.call_args
        self.assertIn("Referer", kwargs["headers"])
        self.assertTrue(kwargs["headers"]["User-Agent"].startswith("Mozilla"))

    @mock.patch("requests.get")
    def test_fetch_share_page_private_chat_raises_share_access_error(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.text = ""
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with self.assertRaises(ShareAccessError) as ctx:
            fetch_share_page("https://chatgpt.com/c/abcdef")

        self.assertIn("private conversation", str(ctx.exception))

    def test_chat_to_markdown_includes_table(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertIn("| Project | Location", markdown)

    def test_chat_markdown_has_no_citation_tokens(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertNotIn("citeturn", markdown)

    def test_chat_markdown_has_no_response_length_blobs(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertNotIn("response_length", markdown)

    def test_chat_markdown_summarizes_tool_queries(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertIn("Search tool invoked with queries:", markdown)

    def test_markdown_preserves_useful_content(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        raw_segments = self._collect_useful_segments(self.html)
        normalized_markdown = self._normalize(markdown)
        for segment in raw_segments:
            with self.subTest(segment=segment):
                self.assertIn(self._normalize(segment), normalized_markdown)

    def test_chat_save_markdown_creates_file(self) -> None:
        chat = parse_modern_share(self.html)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            md_path = chat.save_markdown(path, download_assets=False)
            self.assertTrue(md_path.exists())
            self.assertTrue(md_path.read_text(encoding="utf-8"))

    @mock.patch("ChatPeek.fetch_share_page")
    def test_chatpeek_constructs_chat_once(self, mock_fetch: mock.Mock) -> None:
        mock_fetch.return_value = self.html
        instance = ChatPeek("https://chatgpt.com/share/690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertIsInstance(instance.chat, Chat)
        mock_fetch.assert_called_once()

    def _collect_useful_segments(self, html: str) -> List[str]:
        payload = extract_loader_payload(html)
        self.assertIsNotNone(payload)
        assert payload is not None
        decoded = decode_loader(payload)
        loader_data = cast(Dict[str, Any], decoded.get("loaderData", {}))
        route = cast(Dict[str, Any], loader_data.get("routes/share.$shareId.($action)", {}))
        server_response = cast(Dict[str, Any], route.get("serverResponse", {}))
        data = cast(Dict[str, Any], server_response.get("data", {}))
        mapping = cast(Dict[str, Any], data.get("mapping", {}))
        sequence = cast(List[Dict[str, Any]], data.get("linear_conversation", []))
        segments: List[str] = []

        for entry in sequence:
            node_id = entry.get("id") if isinstance(entry, Mapping) else None
            if not isinstance(node_id, str):
                continue
            node = mapping.get(node_id)
            if not isinstance(node, Mapping):
                continue
            message = node.get("message")
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, Mapping):
                continue
            ctype = content.get("content_type")
            if ctype == "text":
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, str):
                            cleaned = strip_private_use(part).strip()
                            cleaned = self._clean_segment(cleaned)
                            if cleaned:
                                segments.append(cleaned)
            elif ctype == "code":
                text = content.get("text")
                if isinstance(text, str):
                    try:
                        maybe_json = json.loads(text)
                    except json.JSONDecodeError:
                        maybe_json = None
                    if isinstance(maybe_json, Mapping):
                        search_queries = maybe_json.get("search_query")
                        if isinstance(search_queries, list):
                            for entry in search_queries:
                                if isinstance(entry, Mapping):
                                    query = entry.get("q")
                                    if isinstance(query, str) and query.strip():
                                        cleaned_query = self._clean_segment(query.strip())
                                        if cleaned_query:
                                            segments.append(cleaned_query)
                                elif isinstance(entry, str) and entry.strip():
                                    cleaned_entry = self._clean_segment(entry.strip())
                                    if cleaned_entry:
                                        segments.append(cleaned_entry)
                        continue
                    if text.strip():
                        cleaned_text = self._clean_segment(text.strip())
                        if cleaned_text:
                            segments.append(cleaned_text)
            elif ctype == "multimodal_text":
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, str):
                            cleaned = strip_private_use(part).strip()
                            cleaned = self._clean_segment(cleaned)
                            if cleaned:
                                segments.append(cleaned)
                        elif isinstance(part, Mapping):
                            inner_text = part.get("text")
                            if isinstance(inner_text, list):
                                for piece in inner_text:
                                    if isinstance(piece, str):
                                        cleaned_piece = strip_private_use(piece).strip()
                                        if cleaned_piece:
                                            piece_cleaned = self._clean_segment(cleaned_piece)
                                            if piece_cleaned:
                                                segments.append(piece_cleaned)
                            elif isinstance(inner_text, str):
                                cleaned_piece = strip_private_use(inner_text).strip()
                                if cleaned_piece:
                                    piece_cleaned = self._clean_segment(cleaned_piece)
                                    if piece_cleaned:
                                        segments.append(piece_cleaned)
        return segments

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _clean_segment(text: str) -> str:
        cleaned = strip_citation_tokens(text)
        cleaned = re.sub(r"navlist[^\s]*", "", cleaned)
        cleaned = cleaned.replace("citeturn", "")
        cleaned = cleaned.strip()
        return cleaned


if __name__ == "__main__":
    unittest.main()