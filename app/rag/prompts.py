"""The prompts, and the formatting they depend on.

Every rule below was added because a model probed during design broke it.
gemma3:4b answered English questions in Bangla when left to infer the language,
rewrote 6,500 as ৬,৫০০, answered Banglish questions in English when merely told
"answer in Bangla", and appended a bogus "[1]" to refusals -- citing a source
for information it had just said was absent.
"""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.language import AnswerLanguage

# The entire language instruction, per answer language, injected wholesale as
# rule 2. Not a language name interpolated into a shared sentence: naming the
# language was measured and does not work on Banglish input.
LANGUAGE_RULES: dict[AnswerLanguage, str] = {
    # Three things are load-bearing here and none may be trimmed: naming the
    # *script* rather than the language, saying explicitly that a Latin-script
    # question still gets a Bengali-script answer, and the one-shot example.
    # Instruction alone scored 3/4 on Banglish and leaked a Tamil character;
    # with the example, 4/4.
    "Bangla": """Write your ENTIRE answer in Bengali script (বাংলা). Never use \
Latin letters for Bangla words. Even if the question is written with English \
letters (Banglish), you MUST still reply in Bengali script. Use Western digits \
(0-9) for all numbers.

Example:
Question: CSE er total tuition fee koto
Answer: সিএসই টিউশন ফি 700000 BDT [1]।""",
    "English": (
        "Write your ENTIRE answer in English. Use Western digits (0-9) for all numbers."
    ),
}

_ANSWER_SYSTEM = """You are the Dhaka International University admission assistant.

Rules, in priority order:
1. Answer ONLY using the numbered context. Never use outside knowledge.
2. {language_rule}
3. End each sentence you took from the context with its citation, like [1].
4. If the context does not answer the question, reply with exactly this and \
nothing else: {refusal}
5. Never invent a number, date, fee or deadline."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANSWER_SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# Runs before retrieval, so its output becomes the search query. If it answers
# rather than rewrites, retrieval searches for an invented answer.
_CONTEXTUALIZE_SYSTEM = """Rewrite the user's follow-up into a standalone question \
that can be understood without the conversation.

Return ONLY the rewritten question. Do not answer it. Do not explain.
Keep it in the same language and script the user wrote it in.
If it already stands alone, return it unchanged."""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONTEXTUALIZE_SYSTEM),
        ("human", "Conversation so far:\n{history}\n\nFollow-up: {question}"),
    ]
)

# The fallback when the lexicon has no evidence either way. Labels match
# QuestionLanguage exactly, because the result is looked up in ANSWER_LANGUAGE.
_CLASSIFY_SYSTEM = """Classify the user's question into exactly one label:

bangla_script = written in Bengali letters
banglish = Bangla language written with English letters (e.g. "fee koto", "ki lagbe")
english = ordinary English

Reply with the label only."""

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CLASSIFY_SYSTEM), ("human", "{question}")]
)


def format_docs(docs: Sequence[Document]) -> str:
    """Number the chunks from 1, headed by post title.

    The numbering is a contract between three places: what the model is told to
    cite, what this emits, and what the UI renders as sources[n-1].
    """
    return "\n\n".join(
        f"[{index}] {doc.metadata.get('post_title', 'Untitled')}\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def format_history(history: Sequence[tuple[str, str]]) -> str:
    """(role, content) pairs as plain labelled lines."""
    return "\n".join(f"{role}: {content}" for role, content in history)
