import sensenova
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import *
from langchain_openai.chat_models.base import (
    _convert_message_to_dict,
    _convert_delta_to_message_chunk,
    _is_pydantic_class
)
import os 
_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if not role:
        role = "assistant"
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("message", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)


class ChatSense(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="SenseChat-5", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    sense_api_key_id: str = Field(default=None, alias="api_key_id")
    sense_api_key_secret: str = Field(default=None, alias="api_key_secret")
    sense_session_id: str = Field(default=None, alias="session_id")
    system_prompt: str = Field(default="you are a helpful assistant")
    streaming: bool = False
    max_tokens: int = 1024
    """有无对话历史"""
    with_history: bool = True

    n: int = 1
    """Number of chat completions to generate for each prompt."""

    @root_validator(pre=True)
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        if  values["sense_api_key_id"]:
            sensenova.access_key_id = values["sense_api_key_id"]
        if  values["sense_api_key_secret"]:
            sensenova.secret_access_key = values["sense_api_key_secret"]
        # 是否有对话历史
        if values["with_history"]:
            values["client"] = sensenova.ChatSession.create(
                system_prompt=[{"role": "system", "content": values["system_prompt"]}]
            )
            values["sense_session_id"] = values["client"]["session_id"]
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_new_tokens"] = self.max_tokens
        return params

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        if self.with_history:
            session_id = self.client["session_id"]
            params.update({"session_id": session_id})
        response = sensenova.ChatCompletion.create(messages=message_dicts, **params)
        response = response.data.to_dict_recursive()
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await sensenova.ChatCompletion.acreate(
            messages=message_dicts, **params
        )
        response = response.data.to_dict_recursive()
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class = AIMessageChunk
        resp = sensenova.ChatCompletion.create(messages=message_dicts, **params)
        for part in resp:
            part_dict = part.to_dict_recursive()
            chunk = part_dict.get("data", {})
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            choice["content"] = choice.get("delta", "")
            chunk = _convert_delta_to_message_chunk(choice, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        params.update({"stream": True})
        default_chunk_class = AIMessageChunk
        resp = await sensenova.ChatCompletion.acreate(messages=message_dicts, **params)
        for part in resp:
            part_dict = part.to_dict_recursive()
            chunk = part_dict.get("data", {})
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            choice["content"] = choice.get("delta", "")
            chunk = _convert_delta_to_message_chunk(choice, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
                yield chunk

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()

        # Sometimes the AI Model calling will get error, we should raise it.
        # Otherwise, the next code 'choices.extend(response["choices"])'
        # will throw a "TypeError: 'NoneType' object is not iterable" error
        # to mask the true error. Because 'response["choices"]' is None.
        if response.get("error"):
            raise ValueError(response.get("error"))

        for res in response["choices"]:
            message = _convert_dict_to_message(res)
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sense-nova-model"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "manual"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                name of the tool (str): calls corresponding tool;
                "auto": automatically selects a tool (including no tool);
                "manual": select a tool manually
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "manual"):
                    tool_choice = {
                        "mode": "manual",
                        "tools": [tool_choice],
                    }
                elif tool_choice == "auto":
                    tool_choice = {
                        "mode":"auto",
                        "tools":[]
                    }
                elif tool_choice == "manual":
                    if len(formatted_tools)>1:
                        raise ValueError(
                            "Sense model only supports one tool choice if mode='manual' currently."
                        )
                    tool_choice = {
                        "mode":"manual",
                        "tools":[{"type":formatted_tools[0]["type"],"name":formatted_tools[0]["function"]["name"]}]
                    }
            elif isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be used when a single tool is "
                        f"passed in, received {len(tools)} tools."
                    )
                tool_choice = {
                    "mode": "manual",
                    "tools": [{"type":formatted_tools[0]["type"],"name":formatted_tools[0]["function"]["name"]}],
                }
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if len(tool_choice["tools"])>1:
                    raise ValueError(
                        "Sense model only supports one tool choice if mode='manual' currently."
                    )
                if not any(
                    tool_name == tool_choice["tools"][0]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]: ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]: ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then OpenAI's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            llm = self.bind_tools([schema], tool_choice=True)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
    
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_format'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser


if __name__ == "__main__":
    model = ChatSense(
        api_key_id="your api key here, or set it in the environment",
        api_key_secret="your api secret here, or set it in the environment",
        streaming=False
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
    
    # input image 
    model = ChatSense(model="SenseChat-Vision",api_key_id=os.environ['SENSENOVA_ACCESS_KEY_ID'],api_key_secret=os.environ['SENSENOVA_SECRET_ACCESS_KEY'])
    import pickle as pkl 
    content = pkl.load(open(r"F:\projects\reports_llm\industry_report\消费\社会服务.pkl",'rb'))[0]['image']
    model.invoke([HumanMessage(
        content=[
            {"type":"text","text":"描述这张图片"},
            {"type":"image_base64","image_base64":content.decode()}
        ]
    )])