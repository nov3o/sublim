import asyncio
from typing import Literal
from openai.types import FileObject
from sl.llm.data_models import LLMResponse, Chat
from sl import config
from sl.llm.services import SampleCfg
from sl.utils import fn_utils
import openai


_client = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.max_concurrency_async(max_size=1000)
async def sample(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    kwargs = sample_cfg.model_dump()
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]

    api_response = await get_client().chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], model=model_id, **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")

    # Extract logprobs if requested
    parsed_logprobs = None
    sampled_tokens = None
    if sample_cfg.logprobs and choice.logprobs and choice.logprobs.content:
        parsed_logprobs = []
        sampled_tokens = []
        for token_data in choice.logprobs.content:
            token_logprobs = {}
            if token_data.top_logprobs:
                for top_token in token_data.top_logprobs:
                    token_logprobs[top_token.token] = top_token.logprob
            parsed_logprobs.append(token_logprobs)
            # OpenAI provides the actual token in token_data.token
            sampled_tokens.append(token_data.token if hasattr(token_data, 'token') else "")

    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=parsed_logprobs,
        sampled_tokens=sampled_tokens,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    return await asyncio.gather(
        *[sample(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)]
    )


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)
