import requests
import json
from bs4 import BeautifulSoup
from enum import Enum
from typing import List


class ReplyType(Enum):
    HUMAN = 0
    AI = 1


class Reply:
    """
    Reply class represents a reply in the chat.
    """
    def __init__(self, name: str, type: ReplyType, statement: str):
        self.name = name
        self.type = type
        self.statement = statement


class Chat:
    """
    Chat class represents a chat.
    """
    def __init__(self, title: str, replies: List[Reply]):
        self.__title = title
        self.__replies = replies

    @property
    def title(self) -> str:
        return self.__title

    @property
    def conversation(self) -> List[Reply]:
        return self.__replies

    @property
    def human_replies(self) -> List[Reply]:
        return [reply for reply in self.__replies if reply.type == ReplyType.HUMAN]

    @property
    def ai_replies(self) -> List[Reply]:
        return [reply for reply in self.__replies if reply.type == ReplyType.AI]


class ChatPeek:
    """
    ChatPeek class scrapes a chat from a link.
    """
    def __init__(self, link: str):
        self.__link = link
        self.__chat = self.__parse_link()

    @property
    def chat(self) -> Chat:
        return self.__chat

    def __parse_link(self) -> Chat:
        content = self.__download_content()
        title, replies = self.__parse_content(content)
        return Chat(title, replies)

    def __download_content(self) -> str:
        try:
            response = requests.get(self.__link)
            response.raise_for_status()  # Raise an exception if the request was unsuccessful
        except requests.RequestException as e:
            print(f"Failed to download content: {e}")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        return script_tag.string if script_tag else ""

    def __parse_content(self, content: str) -> List[Reply]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return "", []

        title = data['props']['pageProps']['serverResponse']['data'].get('title', "")
        conversation_mapping = data['props']['pageProps']['serverResponse']['data'].get('mapping', {})
        replies = []
        for key, message in conversation_mapping.items():
            try:
                reply_text = ''.join(message['message']['content']['parts'])
                if message['message']['author']['role'] == 'user':
                    replies.append(Reply("User", ReplyType.HUMAN, reply_text))
                elif message['message']['author']['role'] == 'assistant':
                    replies.append(Reply("GPT", ReplyType.AI, reply_text))
            except KeyError:
                continue
        replies = list(reversed(replies))
        return title, replies