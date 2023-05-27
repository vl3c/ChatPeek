import unittest
from ChatPeek import ChatPeek, Chat, Reply

class TestChatPeek(unittest.TestCase):
    def test_chat_peek(self):
        chat_peek = ChatPeek("https://chat.openai.com/share/c51a8a6b-e689-40fc-a90f-a39ba24074b0")  # insert your link here
        chat = chat_peek.chat
        self.assertIsInstance(chat, Chat)
        self.assertIsNotNone(chat.title)
        self.assertGreater(len(chat.conversation), 0)
        for reply in chat.conversation:
            self.assertIsInstance(reply, Reply)
            self.assertIsNotNone(reply.name)
            self.assertIsNotNone(reply.statement)

if __name__ == "__main__":
    unittest.main()