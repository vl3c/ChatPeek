import unittest
from datetime import datetime
from ChatPeek import ChatPeek, Chat, Reply

link = "https://chat.openai.com/share/c51a8a6b-e689-40fc-a90f-a39ba24074b0"
validation_text = """Human vs. Chatbot | 2023-05-27 15:04:10 | text-davinci-002-render-sha
User (HUMAN): I am human. What are you?
GPT3.5 (AI): I am an AI language model called ChatGPT. I'm designed to understand and generate human-like text based on the prompts and questions I receive. While I can simulate conversations and provide information, it's important to note that I don't possess consciousness, emotions, or a physical form. I'm purely a computer program created to assist with generating text-based responses.
"""

class TestChatPeek(unittest.TestCase):
    def test_chat_peek_obj(self):
        chat_peek = ChatPeek(link)  
        chat = chat_peek.chat
        self.assertIsInstance(chat, Chat)
        self.assertIsNotNone(chat.title)
        self.assertGreater(len(chat.conversation), 0)
        for reply in chat.conversation:
            self.assertIsInstance(reply, Reply)
            self.assertIsNotNone(reply.name)
            self.assertIsNotNone(reply.statement)
    
    def test_chat_peek_content(self):
        chat_peek = ChatPeek(link)  
        chat = chat_peek.chat
        all_replies = chat.conversation
        s = f"{chat.title} | {datetime.fromtimestamp(chat.date)} | {chat.ai_model}\n"
        for reply in all_replies:
            s += f"{reply.name} ({str(reply.type).split('.')[1]}): {reply.statement}\n"
        print(s)
        self.assertMultiLineEqual(s, validation_text)

if __name__ == "__main__":
    unittest.main()