# ChatSense: Langchain Chatmodel for SenseNova(商汤大模型)

A class to connect Langchain Chatmodel with SenseNova(商汤大模型), based on langchain_core and langchain_openai.

langchain documents: [Introduction | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/introduction/)

SenseNova is a leading LLM especially efficient for Chinese: [文档中心 | 日日新开放平台 (sensenova.cn)](https://platform.sensenova.cn/doc?path=/chat/GetStarted/Library.md)

The ChatSense class is simply based on the ChatOpenAI and is adapted to fit the SenseNova api.

This model supports basic chat model apis, tools calling and `with_structured_output`

Dependencies:

```
langchain-core == 0.1.52
langchain-community == 0.0.38
langchain-openai == 0.1.7
```

Examples:

```python
from chatsense import ChatSense
model = ChatSense(
        api_key_id="your api key here, or set it in the environment",
        api_key_secret="your api secret here, or set it in the environment",
        streaming=False,
	# whether to use SenseNova session to memory the history,seems to be useless because langchain can memory it for you.
    	with_history: bool = True
    )
# base talking
res = model.invoke("你好")
print(res)

#tools calling 
from langchain_community.tools import MoveFileTool
tools = [MoveFileTool()]
model_with_functions = model.bind_tools(tools)
res = model_with_functions.invoke([HumanMessage(content="move file foo to bar")])
print(res)
# structured output 

class Joke(BaseModel):
    """A Joke Structure"""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

structured_llm = model.with_structured_output(Joke, method="function_calling")
res = structured_llm.invoke(
"Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
)
print(res)
```
