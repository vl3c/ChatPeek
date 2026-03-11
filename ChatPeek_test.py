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
    ConversationAsset,
    JsonValue,
    Reply,
    ReplyType,
    ShareAccessError,
    _extract_scripts,
    author_name_for_role,
    build_asset_filename,
    decode_loader,
    extract_loader_payload,
    fetch_share_page,
    flatten_message_content,
    main,
    parse_legacy_share,
    parse_modern_share,
    parse_share_html,
    slugify_title,
    strip_private_use,
    strip_citation_tokens,
    summarize_tool_payload,
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
        self.assertIn("Thinking: about this problem", text)
        self.assertIn("Considering: alternatives", text)

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
        self.assertIn("Just a summary", text)

    def test_reasoning_recap_content_type(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "reasoning_recap", "content": "Recap of reasoning"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertIn("Recap of reasoning", text)

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
        self.assertEqual(text, "Custom instructions here")

    def test_tool_response_strips_private_use(self) -> None:
        message: Dict[str, Any] = {
            "id": "m1",
            "content": {"content_type": "tool_response", "output": "Result\uE000here"},
            "metadata": {},
        }
        text, _ = flatten_message_content("m1", message["content"], message)
        self.assertEqual(text, "Resulthere")

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


class FetchSharePageTests(unittest.TestCase):
    @mock.patch("requests.get")
    def test_non_403_error_re_raises(self, mock_get: mock.Mock) -> None:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            fetch_share_page("https://chatgpt.com/share/abc")

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
                    url="https://example.com/report.csv",
                    filename="report.csv",
                    downloadable=True,
                )
            ],
        )
        chat = Chat(share_id="abcdef12", ai_model="", title="Test", updated_at=None, replies=[reply])
        mock_response = mock.Mock(spec=requests.Response)
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


if __name__ == "__main__":
    unittest.main()