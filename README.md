# ChatPeek - a ChatGPT shared conversation link parser

ChatPeek is a Python utility for parsing shared conversation links from OpenAI's ChatGPT. It allows users to easily analyze, process and work with conversations.

## Features

- Fetches shared conversation content from a given link.
- Parses the content into a structured format.
- Classifies each message in the conversation as either human or AI.
- Provides easy access to all replies in the conversation, as well as distinguishing between human and AI replies.

## Usage

First, instantiate a `ChatPeek` object with a shared conversation link as an argument. Then, access the `Chat` object via the `chat` property of the `ChatPeek` object:

```python
chat_peek = ChatPeek("your_shared_conversation_link_here") 
chat = chat_peek.chat
```

You can then access the conversation title, all replies, and specific replies:

```python
all_replies = chat.conversation
print(chat.title)
print(f"{datetime.datetime.fromtimestamp(chat.date)} | {chat.ai_model}\n")
for reply in all_replies:
    print(f"{reply.name} ({str(reply.type).split('.')[1]}):\n{reply.statement}\n")
```

## Testing

Unit tests are provided in `ChatPeek_test.py`. To run the tests, use the following command:

```bash
python -m unittest ChatPeek_test.py
```

## Contributing

Contributions are welcome! Please submit a pull request or create an issue to discuss any changes you wish to make.

## License

This project is licensed under the MIT License.

## Classes

### ReplyType
An Enum for specifying the type of reply, either `HUMAN` or `AI`.

### Reply
Represents a single reply in a conversation, with a author names (if public), a reply type, and the statement content.

### Chat
Represents a full chat conversation, with a timestamp, user name, AI model, a title and a list of `Reply` objects.

### ChatPeek
The main class that takes a ChatGPT shared conversation link and parses it into a `Chat` object. 
