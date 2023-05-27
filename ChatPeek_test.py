import unittest
from ChatPeek import ChatPeek, Chat, Reply

class TestChatPeek(unittest.TestCase):
    def test_chat_peek(self):
        chat_peek = ChatPeek("...")  # insert your link here
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