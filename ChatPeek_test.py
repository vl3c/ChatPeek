import json
import os
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast
from unittest import mock

import requests

from check_fixture_links import FIXTURE_LINKS, check_fixture_link

from ChatPeek import (
    Chat,
    ChatPeek,
    ExportOptions,
    ConversationAsset,
    JsonValue,
    Reply,
    ReplyType,
    ShareAccessError,
    _AssetNamer,
    _extract_scripts,
    author_name_for_role,
    build_asset_filename,
    decode_loader,
    default_http_get,
    extract_loader_payload,
    fetch_share_page,
    flatten_message_content,
    is_allowed_asset_url,
    main,
    parse_legacy_share,
    parse_modern_share,
    parse_post_share,
    parse_share_html,
    sanitize_asset_filename,
    slugify_title,
    is_tool_addressed,
    strip_private_use,
    strip_citation_tokens,
    summarize_tool_payload,
)


FIXTURES: Path = Path(__file__).resolve().parent / "fixtures"
SHARE_FIXTURE: Path = FIXTURES / "690781ed-75f0-8006-9d6e-d9229bd932f2.html"
# A varied conversation (many turns, multi-language code blocks, nested
# Markdown) captured as HTML so the tests keep exercising those paths even
# after the live share link expires.
VARIED_FIXTURE: Path = FIXTURES / "69b1c492-1540-8006-aa29-ee2e0a831385.html"


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
        # Default export keeps only user-visible replies (5 user + 4 assistant).
        self.assertEqual(len(chat.replies), 9)
        full = parse_modern_share(
            self.html,
            ExportOptions(
                include_reasoning=True,
                include_tool_output=True,
                include_model_context=True,
            ),
        )
        self.assertGreater(len(full.replies), len(chat.replies))

    def test_parse_post_share_returns_message_slice_and_widget_report(self) -> None:
        report_message = {
            "id": "report",
            "author": {"role": "assistant", "metadata": {}},
            "create_time": 1777135581.0,
            "content": {
                "content_type": "text",
                "parts": ["# Deep Research Report\n\nDetailed findings."],
            },
            "metadata": {},
        }
        html = self._build_loader_html(
            {
                "loaderData": {
                    "routes/s.$postId": {
                        "postWithProfile": {
                            "post": {
                                "id": "t_post123",
                                "posted_at": 1777152234.0,
                                "text": "Post Share",
                                "attachments": [
                                    {
                                        "kind": "message_slice",
                                        "messages": [
                                            {
                                                "id": "user-message",
                                                "author": {"role": "user", "metadata": {}},
                                                "create_time": 1777134625.0,
                                                "content": {
                                                    "content_type": "text",
                                                    "parts": ["Find the implementation"],
                                                },
                                                "metadata": {},
                                            },
                                            {
                                                "id": "tool-message",
                                                "author": {"role": "tool", "metadata": {}},
                                                "create_time": 1777134626.0,
                                                "content": {
                                                    "content_type": "code",
                                                    "language": "json",
                                                    "text": "{\"session_id\":\"abc\"}",
                                                },
                                                "metadata": {
                                                    "chatgpt_sdk": {
                                                        "widget_state": json.dumps(
                                                            {"report_message": report_message}
                                                        )
                                                    }
                                                },
                                            },
                                        ],
                                    }
                                ],
                            }
                        }
                    }
                }
            }
        )

        chat = parse_post_share(html)

        self.assertEqual(chat.share_id, "t_post123")
        self.assertEqual(chat.title, "Post Share")
        self.assertEqual(chat.updated_at, 1777152234.0)
        self.assertEqual([reply.type for reply in chat.replies], [
            ReplyType.HUMAN,
            ReplyType.TOOL,
            ReplyType.AI,
        ])
        self.assertIn("Find the implementation", chat.to_markdown())
        self.assertIn("Deep Research Report", chat.to_markdown())

    def test_parse_post_share_respects_export_options(self) -> None:
        html = self._build_loader_html(
            {
                "loaderData": {
                    "routes/s.$postId": {
                        "postWithProfile": {
                            "post": {
                                "id": "t_post789",
                                "posted_at": 1777152234.0,
                                "text": "Post Share",
                                "attachments": [
                                    {
                                        "kind": "message_slice",
                                        "messages": [
                                            {
                                                "id": "thought-message",
                                                "author": {"role": "assistant", "metadata": {}},
                                                "create_time": 1777134625.0,
                                                "content": {
                                                    "content_type": "thoughts",
                                                    "thoughts": [
                                                        {"summary": "Secret reasoning"}
                                                    ],
                                                },
                                                "metadata": {},
                                            },
                                            {
                                                "id": "tool-call",
                                                "author": {"role": "assistant", "metadata": {}},
                                                "recipient": "web",
                                                "create_time": 1777134626.0,
                                                "content": {
                                                    "content_type": "code",
                                                    "text": 'search("internal query")',
                                                },
                                                "metadata": {},
                                            },
                                            {
                                                "id": "redacted-tool",
                                                "author": {"role": "tool", "metadata": {}},
                                                "create_time": 1777134627.0,
                                                "content": {
                                                    "content_type": "text",
                                                    "parts": [
                                                        "The output of this plugin was redacted."
                                                    ],
                                                },
                                                "metadata": {},
                                            },
                                            {
                                                "id": "answer",
                                                "author": {"role": "assistant", "metadata": {}},
                                                "create_time": 1777134628.0,
                                                "content": {
                                                    "content_type": "text",
                                                    "parts": ["Visible answer"],
                                                },
                                                "metadata": {},
                                            },
                                        ],
                                    }
                                ],
                            }
                        }
                    }
                }
            }
        )

        default_markdown = parse_post_share(html).to_markdown()
        self.assertIn("Visible answer", default_markdown)
        self.assertNotIn("Secret reasoning", default_markdown)
        self.assertNotIn("internal query", default_markdown)
        self.assertNotIn("The output of this plugin was redacted.", default_markdown)

        full_markdown = parse_post_share(
            html,
            ExportOptions(include_reasoning=True, include_tool_output=True),
        ).to_markdown()
        self.assertIn("Secret reasoning", full_markdown)
        self.assertIn("internal query", full_markdown)
        self.assertNotIn("The output of this plugin was redacted.", full_markdown)

    def test_parse_share_html_prefers_post_share_route(self) -> None:
        html = self._build_loader_html(
            {
                "loaderData": {
                    "routes/s.$postId": {
                        "postWithProfile": {
                            "post": {
                                "id": "t_post456",
                                "posted_at": 1777152234.0,
                                "text": "Post Route",
                                "attachments": [],
                            }
                        }
                    }
                }
            }
        )

        chat = parse_share_html(html)

        self.assertEqual(chat.share_id, "t_post456")
        self.assertEqual(chat.title, "Post Route")

    def test_parse_post_share_raises_without_post_route(self) -> None:
        html = self._build_loader_html(
            {"loaderData": {"routes/share.$shareId.($action)": {}}}
        )

        with self.assertRaises(ValueError):
            parse_post_share(html)

    def test_parse_post_share_raises_when_loader_data_not_mapping(self) -> None:
        html = self._build_loader_html({"loaderData": "routes/s.$postId"})

        with self.assertRaises(ValueError):
            parse_post_share(html)

    def test_parse_post_share_tolerates_non_mapping_post(self) -> None:
        html = self._build_loader_html(
            {
                "loaderData": {
                    "routes/s.$postId": {"postWithProfile": {"post": ["bogus"]}}
                }
            }
        )

        chat = parse_post_share(html)

        self.assertEqual(chat.share_id, "shared")
        self.assertEqual(chat.replies, [])

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
                        "download_url": "https://files.oaiusercontent.com/file.txt",
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
                        "asset_pointer": "https://files.oaiusercontent.com/asset.bin",
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

    def test_chat_markdown_omits_internal_reasoning_by_default(self) -> None:
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertNotIn("Thought for a couple of seconds", markdown)
        self.assertNotIn("Search tool invoked with queries:", markdown)
        self.assertNotIn("The output of this plugin was redacted.", markdown)
        # Non-JSON tool invocations (e.g. search("...") addressed to the web
        # tool) must be hidden too, not just JSON payloads.
        self.assertNotIn("search(", markdown)
        self.assertNotIn("search_query", markdown)

    def test_parse_share_html_can_include_internal_content(self) -> None:
        chat = parse_share_html(
            self.html,
            ExportOptions(
                include_reasoning=True,
                include_tool_output=True,
                include_model_context=True,
            ),
        )
        markdown = chat.to_markdown()
        self.assertIn("Thinking longer for a better answer", markdown)
        self.assertIn("Original custom instructions no longer available", markdown)
        self.assertIn("Search tool invoked with queries:", markdown)
        self.assertNotIn("The output of this plugin was redacted.", markdown)
        self.assertIn("Thought for 40s", markdown)

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
            author_info = message.get("author") or {}
            role = author_info.get("role") if isinstance(author_info, Mapping) else None
            ctype = content.get("content_type")
            if role == "tool":
                continue
            if is_tool_addressed(message):
                continue
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

    def _build_loader_html(self, decoded_payload: Mapping[str, Any]) -> str:
        loader: List[JsonValue] = ["root"]
        for key, value in decoded_payload.items():
            loader.extend([key, cast(JsonValue, value)])
        chunk = json.dumps(json.dumps(loader))
        return f"<html><script>streamController.enqueue({chunk});</script></html>"

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


class ExtractScriptsTests(unittest.TestCase):
    def test_extracts_multiple_scripts(self) -> None:
        html = "<html><script>var a=1;</script><script>var b=2;</script></html>"
        scripts = _extract_scripts(html)
        self.assertEqual(len(scripts), 2)
        self.assertEqual(scripts[0][1], "var a=1;")
        self.assertEqual(scripts[1][1], "var b=2;")

    def test_preserves_script_attributes(self) -> None:
        html = '<html><script id="__NEXT_DATA__" type="application/json">{"k":"v"}</script></html>'
        scripts = _extract_scripts(html)
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0][0]["id"], "__NEXT_DATA__")
        self.assertEqual(scripts[0][0]["type"], "application/json")

    def test_empty_script_tag(self) -> None:
        html = "<html><script></script></html>"
        scripts = _extract_scripts(html)
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0][1], "")

    def test_no_scripts(self) -> None:
        html = "<html><body><p>No scripts here</p></body></html>"
        scripts = _extract_scripts(html)
        self.assertEqual(scripts, [])

    def test_script_with_html_entities(self) -> None:
        html = "<html><script>var x = 1 &amp;&amp; 2;</script></html>"
        scripts = _extract_scripts(html)
        self.assertEqual(len(scripts), 1)
        # HTMLParser treats script content as raw text; entities are not decoded
        self.assertEqual(scripts[0][1], "var x = 1 &amp;&amp; 2;")


class ExtractLoaderPayloadTests(unittest.TestCase):
    def test_returns_none_for_html_without_scripts(self) -> None:
        self.assertIsNone(extract_loader_payload("<html><body>hi</body></html>"))

    def test_returns_none_for_script_without_enqueue(self) -> None:
        html = "<html><script>var x = 42;</script></html>"
        self.assertIsNone(extract_loader_payload(html))

    def test_returns_none_for_empty_html(self) -> None:
        self.assertIsNone(extract_loader_payload(""))

    def test_returns_none_for_enqueue_with_non_list_payload(self) -> None:
        html = '<html><script>streamController.enqueue("just a string");</script></html>'
        self.assertIsNone(extract_loader_payload(html))

    def test_extracts_simple_list_payload(self) -> None:
        payload_data = ["hello", 1, {"key": "value"}]
        inner = json.dumps(json.dumps(payload_data))
        html = f"<html><script>streamController.enqueue({inner});</script></html>"
        result = extract_loader_payload(html)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, payload_data)

    def test_skips_non_list_enqueue_finds_list(self) -> None:
        non_list = json.dumps(json.dumps("just a string"))
        payload_data = ["target", 2]
        list_payload = json.dumps(json.dumps(payload_data))
        html = (
            f"<html><script>"
            f"streamController.enqueue({non_list});"
            f"streamController.enqueue({list_payload});"
            f"</script></html>"
        )
        result = extract_loader_payload(html)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, payload_data)

    def test_parenthesized_enqueue_argument(self) -> None:
        html = '<html><script>streamController.enqueue(([1, 2, 3]));</script></html>'
        result = extract_loader_payload(html)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, [1, 2, 3])

    def test_enqueue_with_malformed_json_skips(self) -> None:
        html = (
            "<html><script>"
            'streamController.enqueue("{broken json");'
            "</script></html>"
        )
        self.assertIsNone(extract_loader_payload(html))

    def test_searches_correct_script_tag(self) -> None:
        payload_data = ["found", 42]
        inner = json.dumps(json.dumps(payload_data))
        html = (
            "<html>"
            "<script>var unrelated = true;</script>"
            f"<script>streamController.enqueue({inner});</script>"
            "</html>"
        )
        result = extract_loader_payload(html)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, payload_data)


class DecodeLoaderTests(unittest.TestCase):
    def test_simple_key_value_pairs(self) -> None:
        loader: List[JsonValue] = ["header", "key1", "value1", "key2", "value2"]
        result = decode_loader(loader)
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], "value2")

    def test_resolves_integer_references(self) -> None:
        loader: List[JsonValue] = ["header", "name", 0]
        result = decode_loader(loader)
        self.assertEqual(result["name"], "header")

    def test_odd_length_loader_ignores_trailing(self) -> None:
        loader: List[JsonValue] = ["header", "key1", "value1", "orphan"]
        result = decode_loader(loader)
        self.assertEqual(result["key1"], "value1")
        self.assertNotIn("orphan", result)

    def test_nested_dict_resolution(self) -> None:
        loader: List[JsonValue] = ["header", "data", {"inner": 0}]
        result = decode_loader(loader)
        data = cast(Dict[str, Any], result["data"])
        self.assertEqual(data["inner"], "header")

    def test_nested_list_resolution(self) -> None:
        loader: List[JsonValue] = ["header", "items", [0, 0]]
        result = decode_loader(loader)
        items = cast(List[Any], result["items"])
        self.assertEqual(items, ["header", "header"])

    def test_out_of_bounds_reference_returns_raw(self) -> None:
        loader: List[JsonValue] = ["header", "ref", 999]
        result = decode_loader(loader)
        self.assertEqual(result["ref"], 999)

    def test_decode_key_underscore_prefix(self) -> None:
        loader: List[JsonValue] = ["realkey", "data", {"_0": "value"}]
        result = decode_loader(loader)
        data = cast(Dict[str, Any], result["data"])
        self.assertEqual(data["realkey"], "value")

    def test_duplicate_key_keeps_first(self) -> None:
        loader: List[JsonValue] = ["header", "key", "first", "key", "second"]
        result = decode_loader(loader)
        self.assertEqual(result["key"], "first")

    def test_empty_loader(self) -> None:
        result = decode_loader(["header"])
        self.assertEqual(result, {})

    def test_circular_reference_does_not_crash(self) -> None:
        # Index 1 points to index 2, index 2 points to index 1
        # resolve(2) -> cache[2]=None -> resolve(loader[2]) = resolve(2) -> cache hit -> None
        loader: List[JsonValue] = ["header", "a", 2, "b", 1]
        result = decode_loader(loader)
        # Should resolve without infinite loop (cache breaks cycle with None)
        self.assertIn("a", result)
        self.assertIsNone(result["a"])


class AuthorNameTests(unittest.TestCase):
    def test_user_role(self) -> None:
        self.assertEqual(author_name_for_role("user"), "User")

    def test_tool_role(self) -> None:
        self.assertEqual(author_name_for_role("tool"), "Tool")

    def test_assistant_role(self) -> None:
        self.assertEqual(author_name_for_role("assistant"), "Assistant")

    def test_unknown_role_defaults_to_assistant(self) -> None:
        self.assertEqual(author_name_for_role("unknown"), "Assistant")

    def test_none_role_defaults_to_assistant(self) -> None:
        self.assertEqual(author_name_for_role(None), "Assistant")


class StripPrivateUseTests(unittest.TestCase):
    def test_removes_private_use_characters(self) -> None:
        self.assertEqual(strip_private_use("hello\uE000world"), "helloworld")

    def test_leaves_normal_text_unchanged(self) -> None:
        self.assertEqual(strip_private_use("normal text"), "normal text")

    def test_empty_string(self) -> None:
        self.assertEqual(strip_private_use(""), "")

    def test_multiple_private_use_chars(self) -> None:
        self.assertEqual(strip_private_use("\uE001a\uF000b\uF8FFc"), "abc")


class StripCitationTokenTests(unittest.TestCase):
    def test_strips_citeturn_tokens(self) -> None:
        text = "Some text citeturn0search0 more text"
        result = strip_citation_tokens(text)
        self.assertNotIn("citeturn", result)
        self.assertIn("Some text", result)
        self.assertIn("more text", result)

    def test_strips_navlist_tokens(self) -> None:
        text = "Content navlistitem1 rest"
        result = strip_citation_tokens(text)
        self.assertNotIn("navlist", result)

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(strip_citation_tokens(""), "")

    def test_preserves_normal_text(self) -> None:
        text = "Just normal text with no tokens"
        self.assertEqual(strip_citation_tokens(text), text)

    def test_multiline_strips_per_line(self) -> None:
        text = "line1 citeturn0\nline2 navlistfoo"
        result = strip_citation_tokens(text)
        lines = result.split("\n")
        self.assertNotIn("citeturn", lines[0])
        self.assertNotIn("navlist", lines[1])


class SummarizeToolPayloadTests(unittest.TestCase):
    def test_search_query_with_dicts(self) -> None:
        data: Dict[str, Any] = {
            "search_query": [{"q": "python tutorial"}, {"q": "rust guide"}]
        }
        result = summarize_tool_payload(data)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Search tool invoked with queries:", result)
        self.assertIn("python tutorial", result)
        self.assertIn("rust guide", result)

    def test_search_query_with_strings(self) -> None:
        data: Dict[str, Any] = {"search_query": ["query1", "query2"]}
        result = summarize_tool_payload(data)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("query1", result)

    def test_additional_items_without_search(self) -> None:
        data: Dict[str, Any] = {"tool_name": "calculator", "value": 42}
        result = summarize_tool_payload(data)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Tool parameters:", result)
        self.assertIn("tool_name: calculator", result)

    def test_response_length_excluded(self) -> None:
        data: Dict[str, Any] = {"response_length": 500}
        result = summarize_tool_payload(data)
        self.assertIsNone(result)

    def test_empty_payload(self) -> None:
        self.assertIsNone(summarize_tool_payload({}))

    def test_search_query_with_empty_strings_ignored(self) -> None:
        data: Dict[str, Any] = {"search_query": ["", "  ", "valid"]}
        result = summarize_tool_payload(data)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("valid", result)
        # Only 1 bullet for the valid query
        self.assertEqual(result.count("- "), 1)

    def test_detects_tool_addressed_messages(self) -> None:
        self.assertTrue(is_tool_addressed({"recipient": "web"}))
        self.assertTrue(is_tool_addressed({"recipient": "web.run"}))
        self.assertTrue(is_tool_addressed({"recipient": "python"}))
        self.assertFalse(is_tool_addressed({"recipient": "all"}))
        self.assertFalse(is_tool_addressed({"recipient": ""}))
        self.assertFalse(is_tool_addressed({"recipient": None}))
        self.assertFalse(is_tool_addressed({"recipient": 42}))
        self.assertFalse(is_tool_addressed({}))


class BuildAssetFilenameTests(unittest.TestCase):
    def test_basic_filename(self) -> None:
        self.assertEqual(build_asset_filename("msg-1234", 0, "image/png"), "msg-0.png")

    def test_none_message_id(self) -> None:
        self.assertEqual(build_asset_filename(None, 0, "image/png"), "asset-0.png")

    def test_none_mime_type(self) -> None:
        self.assertEqual(build_asset_filename("msg-123", 0, None), "msg-0.bin")

    def test_no_slash_in_mime(self) -> None:
        self.assertEqual(build_asset_filename("msg-123", 0, "plaintext"), "msg-0.bin")

    def test_index_increments(self) -> None:
        self.assertEqual(build_asset_filename("msg-123", 3, "application/pdf"), "msg-3.pdf")

    def test_message_id_without_dash(self) -> None:
        self.assertEqual(build_asset_filename("abcdef", 0, "image/jpeg"), "abcdef-0.jpeg")

    def test_mime_parameters_stripped_from_extension(self) -> None:
        self.assertEqual(
            build_asset_filename("msg", 0, "image/svg+xml; charset=utf-8"), "msg-0.svg+xml"
        )


class SlugifyTitleTests(unittest.TestCase):
    def test_basic_slugify(self) -> None:
        slug = slugify_title("Hello World", "abcdef12-3456")
        self.assertEqual(slug, "hello-world-abcdef12")

    def test_special_characters_removed(self) -> None:
        slug = slugify_title("Hello! @World# $2024", "abcdef12")
        self.assertNotIn("!", slug)
        self.assertNotIn("@", slug)
        self.assertNotIn("#", slug)

    def test_empty_title_defaults_to_chat(self) -> None:
        slug = slugify_title("", "abcdef12")
        self.assertTrue(slug.startswith("chat"))

    def test_long_title_truncated(self) -> None:
        long_title = "a" * 200
        slug = slugify_title(long_title, "abcdef12")
        # 60 char base + dash + 8 char suffix = 69
        self.assertEqual(len(slug), 69)
        self.assertEqual(slug, "a" * 60 + "-abcdef12")

    def test_unicode_title(self) -> None:
        slug = slugify_title("Héllo Wörld", "abcdef12")
        # Non-ASCII chars are stripped, leaving "h-llo-w-rld"
        self.assertEqual(slug, "h-llo-w-rld-abcdef12")

    def test_all_special_chars_title(self) -> None:
        slug = slugify_title("!@#$%", "abcdef12")
        self.assertTrue(slug.startswith("chat"))
        self.assertIn("abcdef12", slug)


class FlattenMessageContentEdgeCases(unittest.TestCase):
    def test_empty_parts_list(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "text", "parts": []},
            "metadata": {},
        }
        text, assets = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")
        self.assertEqual(assets, [])

    def test_non_string_parts_skipped(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "text", "parts": [42, None, True, "keep"]},
            "metadata": {},
        }
        text, assets = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "keep")

    def test_code_with_unknown_language(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "code", "language": "unknown", "text": "x = 1"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertIn("```\n", text)
        self.assertIn("x = 1", text)

    def test_code_with_response_length_only_json(self) -> None:
        payload = json.dumps({"response_length": 500})
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "code", "language": "json", "text": payload},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

    def test_code_addressed_to_tool_is_omitted_by_default(self) -> None:
        payload = json.dumps({"open": [{"ref_id": "abc"}], "response_length": 500})
        message: Dict[str, Any] = {
            "id": "m1",
            "recipient": "browser",
            "content": {"content_type": "code", "language": "json", "text": payload},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_tool_output=True),
        )
        self.assertIn("open", included_text)

    def test_non_json_code_addressed_to_tool_is_omitted_by_default(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "recipient": "web",
            "content": {"content_type": "code", "text": 'search("secret query")'},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_tool_output=True),
        )
        self.assertIn('search("secret query")', included_text)

    def test_user_visible_json_code_block_is_preserved(self) -> None:
        # A legitimate assistant JSON example whose keys look tool-ish
        # (open, input, file) must survive the export untouched.
        payload = json.dumps({"open": True, "input": "hello", "file": "config.yaml"})
        message: Dict[str, Any] = {
            "id": "m1",
            "recipient": "all",
            "content": {"content_type": "code", "language": "json", "text": payload},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertIn('"input": "hello"', text)
        self.assertIn('"file": "config.yaml"', text)
        self.assertIn("```json", text)

    def test_thoughts_content_type(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "thoughts",
                "thoughts": [
                    {"summary": "Thinking", "content": "about this problem"},
                    {"summary": "Considering", "content": "alternatives"},
                ],
            },
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_reasoning=True),
        )
        self.assertIn("Thinking: about this problem", included_text)
        self.assertIn("Considering: alternatives", included_text)

    def test_thoughts_with_only_summary(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "thoughts",
                "thoughts": [{"summary": "Just a summary"}],
            },
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_reasoning=True),
        )
        self.assertIn("Just a summary", included_text)

    def test_reasoning_recap_content_type(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "reasoning_recap", "content": "Recap of reasoning"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_reasoning=True),
        )
        self.assertIn("Recap of reasoning", included_text)

    def test_reasoning_recap_empty(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "reasoning_recap", "content": ""},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

    def test_model_editable_context(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "model_editable_context", "model_set_context": "Custom instructions here"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_model_context=True),
        )
        self.assertEqual(included_text, "Custom instructions here")

    def test_tool_response_strips_private_use(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "tool_response", "output": "Result\uE000here"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

        included_text, _ = flatten_message_content(
            "m1",
            message["content"],
            message,
            ExportOptions(include_tool_output=True),
        )
        self.assertEqual(included_text, "Resulthere")

    def test_unknown_content_type_with_parts_fallback(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "future_type", "parts": ["fallback text"]},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "fallback text")

    def test_unknown_content_type_without_parts(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "future_type"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "")

    def test_multimodal_text_with_mixed_parts(self) -> None:
        message: Dict[str, Any] = {
            "id": "msg-multi",
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    "text before",
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": "https://example.com/img.png",
                        "mime_type": "image/png",
                    },
                    "text after",
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content("msg-multi", message["content"], message)
        self.assertIn("text before", text)
        self.assertIn("text after", text)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].asset_type, "image")

    def test_multimodal_text_part_with_text_list(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {"content_type": "text", "text": ["line1", "line2"]},
                ],
            },
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertIn("line1", text)
        self.assertIn("line2", text)

    def test_json_part_with_content_fallback(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "text",
                "parts": ['{"content": "fallback value", "other": "stuff"}'],
            },
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "fallback value")

    def test_json_part_without_response_or_content_key(self) -> None:
        raw_json = '{"just": "data", "no": "special keys"}'
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "text",
                "parts": [raw_json],
            },
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        # Original JSON string is preserved as-is
        self.assertEqual(text, raw_json)

    def test_attachment_with_file_url_instead_of_download_url(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "text", "parts": ["See file"]},
            "metadata": {
                "attachments": [
                    {
                        "file_url": "https://example.com/data.csv",
                        "name": "data.csv",
                    }
                ]
            },
        }
        text, assets = flatten_message_content("m1", message["content"], message)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].filename, "data.csv")

    def test_attachment_without_url_skipped(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "text", "parts": ["See file"]},
            "metadata": {
                "attachments": [{"name": "orphan.txt"}]
            },
        }
        _, assets = flatten_message_content("m1", message["content"], message)
        self.assertEqual(len(assets), 0)

    def test_none_message_id_in_flatten(self) -> None:
        message: Dict[str, Any] = {
            "id": None,
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "file",
                        "asset_pointer": "https://example.com/f.pdf",
                        "mime_type": "application/pdf",
                    }
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content(None, message["content"], message)
        self.assertEqual(len(assets), 1)
        self.assertIn("asset-0.pdf", assets[0].filename)


class ChatToMarkdownTests(unittest.TestCase):
    def test_empty_title_uses_default(self) -> None:
        chat = Chat(share_id="abc", ai_model="gpt-4", title="", updated_at=None, replies=[])
        md = chat.to_markdown()
        self.assertIn("# ChatGPT conversation", md)

    def test_no_model_no_timestamp(self) -> None:
        chat = Chat(share_id="abc", ai_model="", title="Test", updated_at=None, replies=[])
        md = chat.to_markdown()
        self.assertIn("# Test", md)
        self.assertNotIn("Model:", md)

    def test_includes_model_and_timestamp(self) -> None:
        chat = Chat(share_id="abc", ai_model="gpt-4o", title="Test", updated_at=1700000000.0, replies=[])
        md = chat.to_markdown()
        self.assertIn("Model: gpt-4o", md)
        self.assertIn("2023", md)

    def test_reply_author_fallback_to_type(self) -> None:
        reply = Reply(author_name="", type=ReplyType.AI, statement="Hello")
        chat = Chat(share_id="abc", ai_model="", title="Test", updated_at=None, replies=[reply])
        md = chat.to_markdown()
        self.assertIn("### Assistant", md)

    def test_reply_statement_stripped(self) -> None:
        reply = Reply(author_name="User", type=ReplyType.HUMAN, statement="  padded  \n\n")
        chat = Chat(share_id="abc", ai_model="", title="Test", updated_at=None, replies=[reply])
        md = chat.to_markdown()
        self.assertIn("padded", md)
        self.assertNotIn("  padded  ", md)

    def test_multiple_replies_in_order(self) -> None:
        replies = [
            Reply(author_name="User", type=ReplyType.HUMAN, statement="Question"),
            Reply(author_name="Assistant", type=ReplyType.AI, statement="Answer"),
        ]
        chat = Chat(share_id="abc", ai_model="", title="Test", updated_at=None, replies=replies)
        md = chat.to_markdown()
        q_pos = md.index("Question")
        a_pos = md.index("Answer")
        self.assertLess(q_pos, a_pos)


class ChatSaveMarkdownTests(unittest.TestCase):
    def test_save_creates_directory_structure(self) -> None:
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://example.com/f.txt",
                    filename="f.txt",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=False)
            self.assertTrue(md_path.exists())
            self.assertIn("test-abcdef12", md_path.parent.name)

    def test_save_without_assets_no_subfolder(self) -> None:
        reply = Reply(author_name="User", type=ReplyType.HUMAN, statement="Hello")
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=False)
            self.assertTrue(md_path.exists())
            self.assertEqual(md_path.parent, Path(tmp))

    def test_save_skips_non_downloadable_assets(self) -> None:
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="image",
                    url="sediment://internal",
                    filename="img.png",
                    downloadable=False,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=True)
            self.assertTrue(md_path.exists())
            # Non-downloadable asset should not create the images directory
            image_file = md_path.parent / "images" / "img.png"
            self.assertFalse(image_file.exists())


class ParseLegacyShareTests(unittest.TestCase):
    def _build_legacy_html(self, data: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {
            "props": {"pageProps": {"serverResponse": {"data": data}}}
        }
        return f"<html><script id='__NEXT_DATA__'>{json.dumps(payload)}</script></html>"

    def test_system_messages_excluded(self) -> None:
        data: Dict[str, Any] = {
            "conversation_id": "test",
            "title": "Test",
            "model": {"slug": "gpt-4"},
            "linear_conversation": [
                {
                    "message": {
                        "id": "sys",
                        "author": {"role": "system"},
                        "content": {"content_type": "text", "parts": ["System prompt"]},
                    }
                },
                {
                    "message": {
                        "id": "user1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                    }
                },
            ],
        }
        chat = parse_legacy_share(self._build_legacy_html(data))
        self.assertEqual(len(chat.replies), 1)
        self.assertEqual(chat.replies[0].statement, "Hello")

    def test_missing_next_data_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_legacy_share("<html><body>no script</body></html>")

    def test_preserves_custom_author_name_for_user(self) -> None:
        data: Dict[str, Any] = {
            "conversation_id": "test",
            "title": "Test",
            "model": {"slug": "gpt-4"},
            "author_name": "Alice",
            "linear_conversation": [
                {
                    "message": {
                        "id": "u1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hi"]},
                    }
                },
            ],
        }
        chat = parse_legacy_share(self._build_legacy_html(data))
        self.assertEqual(chat.replies[0].author_name, "Alice")

    def test_missing_model_slug(self) -> None:
        data: Dict[str, Any] = {
            "conversation_id": "test",
            "title": "Test",
            "model": {},
            "linear_conversation": [],
        }
        chat = parse_legacy_share(self._build_legacy_html(data))
        self.assertEqual(chat.ai_model, "")

    def test_update_time_as_int(self) -> None:
        data: Dict[str, Any] = {
            "conversation_id": "test",
            "title": "Test",
            "model": {"slug": "gpt-4"},
            "update_time": 1700000000,
            "linear_conversation": [],
        }
        chat = parse_legacy_share(self._build_legacy_html(data))
        self.assertEqual(chat.updated_at, 1700000000.0)


class ParseShareHtmlTests(unittest.TestCase):
    def test_falls_back_when_modern_fails(self) -> None:
        legacy_data: Dict[str, Any] = {
            "props": {
                "pageProps": {
                    "serverResponse": {
                        "data": {
                            "conversation_id": "fallback",
                            "title": "Fallback Chat",
                            "model": {"slug": "gpt-3.5"},
                            "linear_conversation": [
                                {
                                    "message": {
                                        "id": "a",
                                        "author": {"role": "user"},
                                        "content": {"content_type": "text", "parts": ["Hi"]},
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
        self.assertEqual(chat.share_id, "fallback")

    def test_raises_when_both_parsers_fail(self) -> None:
        with self.assertRaises(ValueError):
            parse_share_html("<html><body>nothing useful</body></html>")


def _fetch_response(
    status: int,
    text: str = "<html></html>",
    headers: Optional[Dict[str, str]] = None,
) -> mock.Mock:
    """Build a mock Response that fails raise_for_status for non-2xx statuses."""

    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.text = text
    response.headers = headers if headers is not None else {}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


class FetchSharePageTests(unittest.TestCase):
    def setUp(self) -> None:
        # Collect backoff delays instead of sleeping so the suite stays offline
        # and instant.
        self.delays: List[float] = []

    def _sleep(self, seconds: float) -> None:
        self.delays.append(seconds)

    @mock.patch("requests.get")
    def test_non_403_error_re_raises_after_exhausting_attempts(self, mock_get: mock.Mock) -> None:
        mock_get.return_value = _fetch_response(500)

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(self.delays, [5.0, 10.0])

    @mock.patch("requests.get")
    def test_transient_status_retried_then_succeeds(self, mock_get: mock.Mock) -> None:
        mock_get.side_effect = [_fetch_response(503), _fetch_response(200, "<html>ok</html>")]

        html = fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(self.delays, [5.0])

    @mock.patch("requests.get")
    def test_connection_error_retried_then_succeeds(self, mock_get: mock.Mock) -> None:
        mock_get.side_effect = [
            requests.ConnectionError("connection reset"),
            _fetch_response(200, "<html>ok</html>"),
        ]

        html = fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(self.delays, [5.0])

    @mock.patch("requests.get")
    def test_body_read_failure_is_retried(self, mock_get: mock.Mock) -> None:
        # stream=False reads the body inside requests.get, so a connection
        # dropped mid-body raises here rather than surfacing as a status code.
        mock_get.side_effect = [
            requests.exceptions.ChunkedEncodingError("truncated body"),
            _fetch_response(200, "<html>ok</html>"),
        ]

        html = fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(mock_get.call_count, 2)

    @mock.patch("requests.get")
    def test_ssl_error_is_not_retried(self, mock_get: mock.Mock) -> None:
        # A rejected certificate is deterministic: retrying delays the same error.
        mock_get.side_effect = requests.exceptions.SSLError("bad certificate")

        with self.assertRaises(requests.exceptions.SSLError):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.delays, [])

    @mock.patch("requests.get")
    def test_retry_after_overrides_backoff(self, mock_get: mock.Mock) -> None:
        mock_get.side_effect = [
            _fetch_response(429, headers={"Retry-After": "2"}),
            _fetch_response(200, "<html>ok</html>"),
        ]

        html = fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(self.delays, [2.0])

    @mock.patch("requests.get")
    def test_retry_after_is_capped(self, mock_get: mock.Mock) -> None:
        mock_get.return_value = _fetch_response(429, headers={"Retry-After": "86400"})

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(self.delays, [60.0, 60.0])

    @mock.patch("requests.get")
    def test_http_date_retry_after_falls_back_to_backoff(self, mock_get: mock.Mock) -> None:
        # The RFC also permits an HTTP-date, which is not parsed; back off instead.
        mock_get.return_value = _fetch_response(
            503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
        )

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(self.delays, [5.0, 10.0])

    @mock.patch("requests.get")
    def test_timeout_re_raises_after_exhausting_attempts(self, mock_get: mock.Mock) -> None:
        mock_get.side_effect = requests.Timeout("timed out")

        with self.assertRaises(requests.Timeout):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 3)

    @mock.patch("requests.get")
    def test_missing_share_is_not_retried(self, mock_get: mock.Mock) -> None:
        # A 404 is a final answer: the share is gone. Retrying only adds load.
        mock_get.return_value = _fetch_response(404)

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc", sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.delays, [])

    @mock.patch("requests.get")
    def test_private_conversation_is_not_retried(self, mock_get: mock.Mock) -> None:
        mock_get.return_value = _fetch_response(403)

        with self.assertRaises(ShareAccessError):
            fetch_share_page("https://chatgpt.com/c/abcdef", sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.delays, [])

    @mock.patch("requests.get")
    def test_attempts_of_one_disables_retrying(self, mock_get: mock.Mock) -> None:
        mock_get.return_value = _fetch_response(500)

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc", attempts=1, sleep=self._sleep)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(self.delays, [])

    @mock.patch("requests.get")
    def test_403_on_share_url_re_raises_as_http_error(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc")

    @mock.patch("requests.get")
    def test_custom_headers_merged(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response

        fetch_share_page("https://chatgpt.com/share/abc", headers={"X-Custom": "val"})
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["X-Custom"], "val")
        self.assertIn("User-Agent", kwargs["headers"])


class AdditionalEdgeCaseTests(unittest.TestCase):
    def test_multimodal_empty_asset_pointer_skipped(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": "",
                    }
                ],
            },
            "metadata": {},
        }
        text, assets = flatten_message_content("m1", message["content"], message)
        self.assertEqual(len(assets), 0)
        self.assertEqual(text, "")

    def test_code_with_non_string_text(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "code", "language": "python", "text": 42},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertIn("```", text)

    def test_slugify_truncation_strips_trailing_hyphen(self) -> None:
        # 58 a's + "!!" -> slug_base before truncation = "a"*58 + "-"
        # after [:60] = "a"*58 + "-", rstrip("-") = "a"*58
        title = "a" * 58 + "!!"
        slug = slugify_title(title, "abcdef12")
        self.assertFalse(slug.split("-abcdef12")[0].endswith("-"))

    def test_save_markdown_downloads_assets_with_mock_http(self) -> None:
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://files.oaiusercontent.com/report.csv",
                    filename="report.csv",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.content = b"col1,col2\na,b\n"
        mock_response.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(
                Path(tmp),
                download_assets=True,
                http_get=lambda url: mock_response,
            )
            asset_path = md_path.parent / "attachments" / "report.csv"
            self.assertTrue(asset_path.exists())
            self.assertEqual(asset_path.read_bytes(), b"col1,col2\na,b\n")

    @mock.patch("requests.get")
    def test_fetch_403_on_non_chatgpt_domain_raises_http_error(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://example.com/c/abcdef")

    def test_strip_citation_tokens_bare_turn_token(self) -> None:
        text = "Some text turn0search0 and more"
        result = strip_citation_tokens(text)
        self.assertNotIn("turn0search0", result)
        self.assertIn("Some text", result)

    def test_summarize_tool_payload_mixed_dict_and_string_queries(self) -> None:
        data: Dict[str, Any] = {
            "search_query": [{"q": "first query"}, "second query"]
        }
        result = summarize_tool_payload(data)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("first query", result)
        self.assertIn("second query", result)
        self.assertEqual(result.count("- "), 2)

    @mock.patch("ChatPeek.fetch_share_page")
    def test_main_with_skip_assets(self, mock_fetch: mock.Mock) -> None:
        fixture_html = SHARE_FIXTURE.read_text(encoding="utf-8")
        mock_fetch.return_value = fixture_html
        with tempfile.TemporaryDirectory() as tmp:
            main(["https://chatgpt.com/share/abc", "--output", tmp, "--skip-assets"])
            md_files = list(Path(tmp).glob("*.md"))
            self.assertEqual(len(md_files), 1)
            content = md_files[0].read_text(encoding="utf-8")
            self.assertIn("Gigawatt", content)

    @mock.patch("ChatPeek.fetch_share_page")
    def test_main_forwards_attempts(self, mock_fetch: mock.Mock) -> None:
        mock_fetch.return_value = SHARE_FIXTURE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            main(["https://chatgpt.com/share/abc", "--output", tmp, "--skip-assets", "--attempts", "7"])
        self.assertEqual(mock_fetch.call_args.kwargs["attempts"], 7)


class VariedShareFixtureTests(unittest.TestCase):
    """Coverage over a share export whose content is deliberately varied:
    many alternating turns, multi-language code blocks, and Markdown nested
    inside assistant messages."""

    def setUp(self) -> None:
        self.html: str = VARIED_FIXTURE.read_text(encoding="utf-8")

    def test_dispatcher_routes_to_modern_share(self) -> None:
        chat = parse_share_html(self.html)
        self.assertEqual(chat.share_id, "69b1c492-1540-8006-aa29-ee2e0a831385")
        self.assertEqual(chat.title, "Hybrid Cognitive Systems")

    def test_alternating_user_and_assistant_turns(self) -> None:
        chat = parse_modern_share(self.html)
        self.assertEqual(len(chat.replies), 15)
        roles = [reply.type for reply in chat.replies]
        self.assertEqual(roles.count(ReplyType.HUMAN), 8)
        self.assertEqual(roles.count(ReplyType.AI), 7)

    def test_preserves_multi_language_code_blocks(self) -> None:
        markdown = parse_modern_share(self.html).to_markdown()
        self.assertIn("```cpp", markdown)
        self.assertIn("#include <iostream>", markdown)
        self.assertIn("```html", markdown)
        self.assertIn("```md", markdown)

    def test_preserves_markdown_nested_in_assistant_message(self) -> None:
        markdown = parse_modern_share(self.html).to_markdown()
        self.assertIn("# Heading 1", markdown)

    def test_markdown_has_no_citation_tokens(self) -> None:
        markdown = parse_modern_share(self.html).to_markdown()
        self.assertNotIn("citeturn", markdown)

    def test_save_markdown_uses_title_and_share_id_for_filename(self) -> None:
        chat = parse_modern_share(self.html)
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=False)
            self.assertTrue(md_path.exists())
            self.assertEqual(md_path.name, "hybrid-cognitive-systems-69b1c492.md")


@unittest.skipUnless(
    os.environ.get("CHATPEEK_CHECK_LINKS"),
    "network liveness check; set CHATPEEK_CHECK_LINKS=1 to run",
)
class FixtureLinkLivenessTests(unittest.TestCase):
    """Opt-in network check: warn and fail when a fixture's backing share link
    has gone stale, so a maintainer is told to mint a replacement. Skipped by
    default so the normal suite stays offline and deterministic."""

    def test_backing_share_links_are_live(self) -> None:
        problems: List[str] = []
        for link in FIXTURE_LINKS:
            result = check_fixture_link(link)
            if result.message is not None:
                warnings.warn(result.message, stacklevel=2)
            # Only a genuinely stale link is a failure; inconclusive checks
            # (network error, rate limit, bot challenge) must not fail the run.
            if result.status == "stale":
                problems.append(result.message or link.fixture)
        if problems:
            self.fail("\n\n".join(problems))


class IsAllowedAssetUrlTests(unittest.TestCase):
    def test_allows_oaiusercontent_subdomain(self) -> None:
        self.assertTrue(is_allowed_asset_url("https://files.oaiusercontent.com/a/b.png"))

    def test_allows_chatgpt_host(self) -> None:
        self.assertTrue(is_allowed_asset_url("https://chatgpt.com/backend-api/file"))

    def test_allows_dalle_blob_host(self) -> None:
        self.assertTrue(
            is_allowed_asset_url("https://oaidalleapiprodscus.blob.core.windows.net/x/img.png")
        )

    def test_allows_fully_qualified_trailing_dot_host(self) -> None:
        self.assertTrue(is_allowed_asset_url("https://files.oaiusercontent.com./x"))

    def test_rejects_unrelated_host(self) -> None:
        self.assertFalse(is_allowed_asset_url("https://example.com/file.txt"))

    def test_rejects_http_scheme(self) -> None:
        # https only — an allowed host over cleartext http must be refused.
        self.assertFalse(is_allowed_asset_url("http://files.oaiusercontent.com/x"))
        self.assertFalse(is_allowed_asset_url("http://chatgpt.com/x"))

    def test_rejects_non_string(self) -> None:
        self.assertFalse(is_allowed_asset_url(None))
        self.assertFalse(is_allowed_asset_url(123))

    def test_rejects_lookalike_hosts(self) -> None:
        self.assertFalse(is_allowed_asset_url("https://evil-oaiusercontent.com/x"))
        self.assertFalse(is_allowed_asset_url("https://oaiusercontent.com.evil.com/x"))

    def test_rejects_internal_addresses(self) -> None:
        self.assertFalse(is_allowed_asset_url("http://127.0.0.1:8080/admin"))
        self.assertFalse(is_allowed_asset_url("http://169.254.169.254/latest/meta-data/"))

    def test_rejects_non_http_schemes(self) -> None:
        self.assertFalse(is_allowed_asset_url("file:///etc/passwd"))
        self.assertFalse(is_allowed_asset_url("ftp://openai.com/x"))
        self.assertFalse(is_allowed_asset_url("sediment://file_123"))

    def test_rejects_empty_url(self) -> None:
        self.assertFalse(is_allowed_asset_url(""))


class SanitizeAssetFilenameTests(unittest.TestCase):
    def test_plain_name_kept(self) -> None:
        self.assertEqual(sanitize_asset_filename("report.csv", "msg-1", 0, None), "report.csv")

    def test_traversal_reduced_to_basename(self) -> None:
        self.assertEqual(sanitize_asset_filename("../../evil.txt", "msg-1", 0, None), "evil.txt")

    def test_windows_separators_reduced_to_basename(self) -> None:
        self.assertEqual(sanitize_asset_filename("..\\..\\evil.txt", "msg-1", 0, None), "evil.txt")

    def test_absolute_path_reduced_to_basename(self) -> None:
        self.assertEqual(sanitize_asset_filename("/etc/passwd", "msg-1", 0, None), "passwd")

    def test_dot_dot_only_falls_back(self) -> None:
        result = sanitize_asset_filename("../..", "msg-1", 0, "text/plain")
        self.assertEqual(result, build_asset_filename("msg-1", 0, "text/plain"))

    def test_none_falls_back(self) -> None:
        result = sanitize_asset_filename(None, "msg-1", 0, "image/png")
        self.assertEqual(result, "msg-0.png")

    def test_control_and_reserved_chars_replaced(self) -> None:
        result = sanitize_asset_filename('a*b?c"d.txt', "msg-1", 0, None)
        self.assertEqual(result, "a_b_c_d.txt")

    def test_drive_relative_path_reduced_to_basename(self) -> None:
        # "C:evil.txt" is a drive-relative path on Windows; only "evil.txt" is safe.
        self.assertEqual(sanitize_asset_filename("C:evil.txt", "msg-1", 0, None), "evil.txt")

    def test_unc_path_reduced_to_basename(self) -> None:
        self.assertEqual(
            sanitize_asset_filename(r"\\host\share\evil.txt", "msg-1", 0, None), "evil.txt"
        )

    def test_trailing_dots_and_spaces_trimmed(self) -> None:
        # Windows strips trailing dots/spaces at file creation, so they must not
        # survive here or the on-disk name diverges from the Markdown link.
        self.assertEqual(sanitize_asset_filename("report.csv. .", "msg-1", 0, None), "report.csv")

    def test_leading_dot_preserved(self) -> None:
        self.assertEqual(sanitize_asset_filename(".env", "msg-1", 0, None), ".env")

    def test_windows_reserved_name_is_prefixed(self) -> None:
        # Reserved device names are prefixed (not discarded), staying writable on
        # every platform while preserving the user's filename.
        self.assertEqual(sanitize_asset_filename("NUL", "msg-1", 3, None), "_NUL")
        self.assertEqual(sanitize_asset_filename("con", "msg-1", 3, None), "_con")
        self.assertEqual(sanitize_asset_filename("NUL.txt", "msg-1", 3, None), "_NUL.txt")
        self.assertEqual(sanitize_asset_filename("COM1.log", "msg-1", 3, None), "_COM1.log")
        # The prefixed form is no longer reserved.
        self.assertNotIn("_NUL".split(".")[0].upper(), {"NUL"})

    def test_overlong_name_truncated(self) -> None:
        result = sanitize_asset_filename("a" * 5000 + ".txt", "msg-1", 0, None)
        self.assertLessEqual(len(result.encode("utf-8")), 200)
        self.assertTrue(result.endswith(".txt"))

    def test_overlong_multibyte_name_truncated_by_bytes(self) -> None:
        # A CJK name can be <=200 chars yet >255 UTF-8 bytes; the cap is on bytes.
        result = sanitize_asset_filename("あ" * 300 + ".txt", "msg-1", 0, None)
        self.assertLessEqual(len(result.encode("utf-8")), 200)
        self.assertTrue(result.endswith(".txt"))
        # Truncation must not leave a broken partial character.
        result.encode("utf-8").decode("utf-8")


class AssetSecurityTests(unittest.TestCase):
    def test_traversal_attachment_name_is_sanitized_at_parse(self) -> None:
        message: Dict[str, Any] = {
            "id": "msg-evil",
            "content": {"content_type": "text", "parts": ["See file"]},
            "metadata": {
                "attachments": [
                    {
                        "download_url": "https://files.oaiusercontent.com/file.txt",
                        "name": "../../evil.txt",
                        "mime_type": "text/plain",
                    }
                ]
            },
        }
        _, assets = flatten_message_content("msg-evil", message["content"], message)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].filename, "evil.txt")

    def test_off_domain_attachment_not_downloadable(self) -> None:
        message: Dict[str, Any] = {
            "id": "msg-ssrf",
            "content": {"content_type": "text", "parts": ["See file"]},
            "metadata": {
                "attachments": [
                    {
                        "download_url": "http://169.254.169.254/latest/meta-data/",
                        "name": "meta.txt",
                    }
                ]
            },
        }
        text, assets = flatten_message_content("msg-ssrf", message["content"], message)
        self.assertEqual(len(assets), 1)
        self.assertFalse(assets[0].downloadable)
        self.assertIn("not included in export", text)

    def test_save_markdown_refuses_traversal_filename(self) -> None:
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://files.oaiusercontent.com/file.txt",
                    filename="../../evil.txt",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        http_get = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=True, http_get=http_get)
            self.assertTrue(md_path.exists())
            http_get.assert_not_called()
            # The traversal target (two levels above attachments/) must not exist.
            self.assertFalse((Path(tmp) / "evil.txt").exists())
            self.assertFalse((md_path.parent / "evil.txt").exists())

    def test_save_markdown_skips_off_domain_url(self) -> None:
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://example.com/report.csv",
                    filename="report.csv",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        http_get = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=True, http_get=http_get)
            http_get.assert_not_called()
            self.assertFalse((md_path.parent / "attachments" / "report.csv").exists())

    def test_save_markdown_skips_redirect_response(self) -> None:
        # An allowed host that responds with a redirect (redirects are disabled,
        # so this surfaces as a non-2xx status) must not have its body written.
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://files.oaiusercontent.com/report.csv",
                    filename="report.csv",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        redirect = mock.Mock(spec=requests.Response)
        redirect.status_code = 302
        redirect.content = b"<html>redirecting to http://169.254.169.254/</html>"
        redirect.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(
                Path(tmp), download_assets=True, http_get=lambda url: redirect
            )
            self.assertFalse((md_path.parent / "attachments" / "report.csv").exists())

    @mock.patch("requests.get")
    def test_default_http_get_does_not_auto_follow_redirects(self, mock_get: mock.Mock) -> None:
        ok = mock.Mock(spec=requests.Response)
        ok.is_redirect = False
        mock_get.return_value = ok
        default_http_get("https://files.oaiusercontent.com/x")
        # Redirects are resolved manually, never by requests' auto-follow.
        self.assertFalse(mock_get.call_args.kwargs["allow_redirects"])

    @mock.patch("requests.get")
    def test_default_http_get_follows_allowed_redirect(self, mock_get: mock.Mock) -> None:
        # 302 from an allowed host to another allowed host is followed to the 200.
        redirect = mock.Mock(spec=requests.Response)
        redirect.is_redirect = True
        redirect.headers = {"Location": "https://oaidalleapiprodscus.blob.core.windows.net/img.png"}
        final = mock.Mock(spec=requests.Response)
        final.is_redirect = False
        final.status_code = 200
        mock_get.side_effect = [redirect, final]
        result = default_http_get("https://chatgpt.com/backend-api/files/x/download")
        self.assertIs(result, final)
        self.assertEqual(mock_get.call_count, 2)

    @mock.patch("requests.get")
    def test_default_http_get_refuses_off_allowlist_redirect(self, mock_get: mock.Mock) -> None:
        # 302 toward an internal address is NOT followed — the request to the
        # internal host is never made, and the redirect response is returned as-is.
        redirect = mock.Mock(spec=requests.Response)
        redirect.is_redirect = True
        redirect.status_code = 302
        redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        mock_get.return_value = redirect
        result = default_http_get("https://chatgpt.com/backend-api/files/x/download")
        self.assertIs(result, redirect)
        self.assertEqual(mock_get.call_count, 1)  # internal host never requested

    @mock.patch("requests.get")
    def test_default_http_get_bounds_redirect_chain(self, mock_get: mock.Mock) -> None:
        # An endless chain of allowed-host redirects stops after the hop cap
        # instead of looping forever.
        redirect = mock.Mock(spec=requests.Response)
        redirect.is_redirect = True
        redirect.headers = {"Location": "https://files.oaiusercontent.com/next"}
        mock_get.return_value = redirect
        result = default_http_get("https://files.oaiusercontent.com/start")
        self.assertIs(result, redirect)
        # 1 initial GET + at most 5 followed hops.
        self.assertLessEqual(mock_get.call_count, 6)

    def test_malformed_host_does_not_abort_export(self) -> None:
        # "https://.openai.com/x" passes the allowlist but requests raises
        # InvalidURL; the export must still complete instead of crashing.
        def boom(url: str) -> requests.Response:
            raise requests.exceptions.InvalidURL("URL has an invalid label")

        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="file",
                    url="https://.openai.com/x",
                    filename="x.bin",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=True, http_get=boom)
            self.assertTrue(md_path.exists())
            self.assertFalse((md_path.parent / "attachments" / "x.bin").exists())

    def test_save_markdown_downloads_image_asset_into_images_dir(self) -> None:
        # Covers the asset_type="image" branch: routing to images/ and the
        # resolved_parents[True] containment key.
        reply = Reply(
            author_name="User",
            type=ReplyType.HUMAN,
            statement="Hello",
            assets=[
                ConversationAsset(
                    asset_type="image",
                    url="https://files.oaiusercontent.com/pic.png",
                    filename="pic.png",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        resp = mock.Mock(spec=requests.Response)
        resp.status_code = 200
        resp.content = b"PNGDATA"
        resp.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            md_path = chat.save_markdown(Path(tmp), download_assets=True, http_get=lambda url: resp)
            self.assertTrue((md_path.parent / "images" / "pic.png").exists())
            self.assertFalse((md_path.parent / "attachments" / "pic.png").exists())

    def test_save_markdown_refuses_absolute_and_drive_filenames(self) -> None:
        # The containment guard must skip absolute/drive/UNC filenames BEFORE the
        # fetch. A would-succeed 200 response is used so that if the guard were
        # removed, the code would reach http_get — assert_not_called then fails.
        for bad in ("C:\\evil.txt", "\\\\host\\share\\evil.txt", "/etc/evil.txt"):
            reply = Reply(
                author_name="User",
                type=ReplyType.HUMAN,
                statement="Hello",
                assets=[
                    ConversationAsset(
                        asset_type="file",
                        url="https://files.oaiusercontent.com/x",
                        filename=bad,
                        downloadable=True,
                    )
                ],
            )
            chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
            resp = mock.Mock(spec=requests.Response)
            resp.status_code = 200
            resp.content = b"OWNED"
            resp.raise_for_status.return_value = None
            http_get = mock.Mock(return_value=resp)
            with tempfile.TemporaryDirectory() as tmp:
                chat.save_markdown(Path(tmp), download_assets=True, http_get=http_get)
                # Guard fired before any fetch, so the escaping path was never written.
                http_get.assert_not_called()


class AssetNamerTests(unittest.TestCase):
    def test_same_url_reuses_one_filename(self) -> None:
        namer = _AssetNamer()
        first = namer.assign("https://x/a", "report.csv")
        second = namer.assign("https://x/a", "report.csv")
        self.assertEqual(first, second)

    def test_distinct_urls_same_name_disambiguated(self) -> None:
        namer = _AssetNamer()
        self.assertEqual(namer.assign("https://x/1", "report.csv"), "report.csv")
        self.assertEqual(namer.assign("https://x/2", "report.csv"), "report-1.csv")
        self.assertEqual(namer.assign("https://x/3", "report.csv"), "report-2.csv")

    def test_disambiguation_without_extension(self) -> None:
        namer = _AssetNamer()
        self.assertEqual(namer.assign("https://x/1", "data"), "data")
        self.assertEqual(namer.assign("https://x/2", "data"), "data-1")

    def test_colliding_attachment_names_stay_distinct_in_export(self) -> None:
        # Two attachments with the same name but different URLs must both be
        # downloaded to distinct files, with distinct Markdown links.
        namer = _AssetNamer()
        message_a: Dict[str, Any] = {
            "id": "m-a",
            "content": {"content_type": "text", "parts": ["A"]},
            "metadata": {"attachments": [
                {"download_url": "https://files.oaiusercontent.com/one", "name": "report.csv"}
            ]},
        }
        message_b: Dict[str, Any] = {
            "id": "m-b",
            "content": {"content_type": "text", "parts": ["B"]},
            "metadata": {"attachments": [
                {"download_url": "https://files.oaiusercontent.com/two", "name": "report.csv"}
            ]},
        }
        _, assets_a = flatten_message_content("m-a", message_a["content"], message_a, namer=namer)
        _, assets_b = flatten_message_content("m-b", message_b["content"], message_b, namer=namer)
        self.assertNotEqual(assets_a[0].filename, assets_b[0].filename)


if __name__ == "__main__":
    unittest.main()
