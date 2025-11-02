import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from ChatPeek import (
    Chat,
    ChatPeek,
    decode_loader,
    extract_loader_payload,
    fetch_share_page,
    flatten_message_content,
    parse_modern_share,
    parse_share_html,
    slugify_title,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SHARE_FIXTURE = FIXTURES / "690781ed-75f0-8006-9d6e-d9229bd932f2.html"


class ChatPeekModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = SHARE_FIXTURE.read_text(encoding="utf-8")

    def test_extract_loader_payload(self):
        payload = extract_loader_payload(self.html)
        self.assertIsInstance(payload, list)
        self.assertGreater(len(payload), 1000)

    def test_decode_loader_contains_share_id(self):
        loader = extract_loader_payload(self.html)
        decoded = decode_loader(loader)
        route = decoded["loaderData"]["routes/share.$shareId.($action)"]
        self.assertEqual(route["sharedConversationId"], "690781ed-75f0-8006-9d6e-d9229bd932f2")

    def test_parse_modern_share_returns_chat(self):
        chat = parse_modern_share(self.html)
        self.assertIsInstance(chat, Chat)
        self.assertEqual(chat.share_id, "690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertIn("Gigawatt", chat.title)
        self.assertGreater(len(chat.replies), 10)

    def test_flatten_message_content_formats_text(self):
        message = {
            "id": "abc",
            "content": {"content_type": "text", "parts": ["Hello", "World"]},
            "metadata": {},
        }
        text, assets = flatten_message_content("abc", message["content"], message)
        self.assertEqual(text, "Hello\n\nWorld")
        self.assertEqual(assets, [])

    def test_flatten_message_content_with_code(self):
        message = {
            "id": "code",
            "content": {"content_type": "code", "language": "python", "text": "print('hi')\n"},
            "metadata": {},
        }
        text, _ = flatten_message_content("code", message["content"], message)
        self.assertIn("```python", text)
        self.assertIn("print('hi')", text)

    def test_flatten_message_content_adds_attachments(self):
        message = {
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

    def test_flatten_message_content_parses_structured_json(self):
        message = {
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

    def test_flatten_message_content_multimodal_file_pointer(self):
        message = {
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

    def test_slugify_title_includes_share_suffix(self):
        slug = slugify_title("Gigawatt Data Centers", "690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertTrue(slug.startswith("gigawatt-data-centers"))
        self.assertTrue(slug.endswith("690781ed"))

    def test_parse_share_html_falls_back_to_legacy(self):
        legacy_data = {
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
    def test_fetch_share_page_sets_headers(self, mock_get):
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html></html>"
        mock_get.return_value = mock_response
        fetch_share_page("https://chatgpt.com/share/abc")
        args, kwargs = mock_get.call_args
        self.assertIn("Referer", kwargs["headers"])
        self.assertTrue(kwargs["headers"]["User-Agent"].startswith("Mozilla"))

    def test_chat_to_markdown_includes_table(self):
        chat = parse_modern_share(self.html)
        markdown = chat.to_markdown()
        self.assertIn("| Project | Location", markdown)

    def test_chat_save_markdown_creates_file(self):
        chat = parse_modern_share(self.html)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            md_path = chat.save_markdown(path, download_assets=False)
            self.assertTrue(md_path.exists())
            self.assertTrue(md_path.read_text(encoding="utf-8"))

    @mock.patch("ChatPeek.fetch_share_page")
    def test_chatpeek_constructs_chat_once(self, mock_fetch):
        mock_fetch.return_value = self.html
        instance = ChatPeek("https://chatgpt.com/share/690781ed-75f0-8006-9d6e-d9229bd932f2")
        self.assertIsInstance(instance.chat, Chat)
        mock_fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()