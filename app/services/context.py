from app.db.data import STATIC_TIPS
from app.core.config import openai_client as client, get_embeddings
from typing import List, Dict, Tuple
import numpy as np
import faiss

_tip_embeddings: np.ndarray = get_embeddings(STATIC_TIPS)
_dim = _tip_embeddings.shape[1]

_index = faiss.IndexFlatL2(_dim)
_index.add(_tip_embeddings)
_tip_norms = np.linalg.norm(_tip_embeddings, axis=1, keepdims=True)
_normed_tips = _tip_embeddings / (_tip_norms + 1e-12)

def retrieve_tips(metrics: Dict[str, float], k: int = 3) -> List[str]:
    """
    Given average metrics, embed a short query and return the top-k tips.
    """
    query = (
        f"Tips for knee {metrics['knee_angle']:.1f}° "
        f"and trunk {metrics['trunk_angle']:.1f}°"
    )
    q_emb = get_embeddings([query])
    _, idxs = _index.search(q_emb, k)
    return [STATIC_TIPS[i] for i in idxs[0]]

def retrieve_tips_with_scores(
    query: str,
    k: int = 3
) -> List[Tuple[str, float]]:
    """
    Returns the top-k (tip, cosine_similarity) pairs.
    """
    q_emb = get_embeddings([query])
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_normed = q_emb / (q_norm + 1e-12)

    # cosine similarities via dot with normalized tips
    sims = (q_normed @ _normed_tips.T).flatten()

    idxs = np.argsort(-sims)[:k]
    return [(STATIC_TIPS[i], float(sims[i])) for i in idxs]

def build_rag_messages(
    user_messages: List[Dict[str, str]],
    tips: List[str],
    *,
    system_prompt: str,
    prefix: str = "Here are some relevant tips:\n",
    suffix: str = ""
) -> List[Dict[str, str]]:
    """
    Prepend a system prompt + an assistant message containing the tips,
    then the original user messages.

    user_messages: e.g. [{"role":"user","content":"…"}] (can also include prior assistant messages)
    tips: list of strings to include
    system_prompt: first system message to set the behavior
    prefix: textual intro before listing tips
    suffix: extra text after tips (e.g. metrics block)
    """
    context_block = prefix + "\n".join(f"- {t}" for t in tips) + suffix

    rag_messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "assistant", "content": context_block}
    ]
    # then append whatever conversation history you want
    rag_messages.extend(user_messages)
    return rag_messages

