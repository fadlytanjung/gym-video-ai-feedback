from typing import AsyncGenerator, List, Dict, Optional
from anyio import create_task_group, create_memory_object_stream
from anyio.to_thread import run_sync
from anyio import create_task_group, create_memory_object_stream, WouldBlock
from app.core.config import openai_client as client
from app.db.data     import STATIC_TIPS
from app.core.config import get_embeddings
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss
import time

class StreamHelper:
    @staticmethod
    async def wrap_blocking_stream(fn, *args, **kwargs):
        send_chan, recv_chan = create_memory_object_stream[str](max_buffer_size=1000)

        def blocking_runner():
            try:
                for chunk in fn(*args, **kwargs):
                    if not chunk:
                        continue
                    while True:
                        try:
                            send_chan.send_nowait(chunk)
                            break
                        except WouldBlock:
                            time.sleep(0.01)
            finally:
                send_chan.close()

        async with recv_chan:
            async with create_task_group() as tg:
                tg.start_soon(run_sync, blocking_runner)
                async for item in recv_chan:
                    yield item

async def generate_stream(
    user_history: List[Dict[str, str]],
    tips: Optional[List[str]] = None,
    *,
    rag_system_prompt: str,
    rag_prefix: str,
    rag_suffix: str,
    chat_system_prompt: str
) -> AsyncGenerator[str, None]:
    if tips:
        context_block = rag_prefix + "\n".join(f"- {t}" for t in tips) + rag_suffix
        messages = [
            {"role": "system",    "content": rag_system_prompt},
            {"role": "assistant", "content": context_block}
        ] + user_history
    else:
        messages = [{"role": "system", "content": chat_system_prompt}] + user_history

    def blocking_stream():
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            stream=True
        )
        for chunk in resp:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    async for text in StreamHelper.wrap_blocking_stream(blocking_stream):
        yield text
