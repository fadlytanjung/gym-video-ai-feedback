from langchain.memory import ConversationBufferMemory
from langchain.memory import RedisChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage
from app.core.config import openai_client as client
from app.core.config import REDIS_URL

def get_memory(session_id: str) -> ConversationBufferMemory:
    """
    Returns a ConversationBufferMemory backed by Redis for this session_id.
    """
    chat_history = RedisChatMessageHistory(
        redis_url=REDIS_URL,
        session_id=session_id
    )
    return ConversationBufferMemory(chat_memory=chat_history, return_messages=True)

async def run_conversation(
    user_input: str,
    memory
) -> str:
    """
    Run a conversation turn through LangChain's ConversationChain,
    returning the assistant's reply.
    """
    llm = ChatOpenAI(
        client=client,
        model_name="gpt-4o-mini",
        temperature=0.7,
        streaming=False
    )
    chain = ConversationChain(llm=llm, memory=memory)
    response = await chain.apredict(input=user_input)
    return response