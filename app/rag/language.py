"""What language to answer in, and what to say when there is no answer.

Decided here rather than by the model, because the model gets it wrong. Probing
during design: gemma3:4b answered English questions in Bangla when left to
infer, and Qwen3 answered Bangla questions by echoing the English context.

Script detection alone is also wrong. Banglish -- Bangla written in Latin
letters, "CSE er tuition fee koto" -- is how a large share of applicants type,
and a script check calls it English. langdetect was measured on this and scored
0/8, labelling Banglish questions Dutch, Norwegian, Slovenian, Albanian,
Afrikaans and French. General language-ID is script-oriented and romanized
Bangla is outside its distribution, so there is no library to reach for.
"""

import re
from typing import Literal

QuestionLanguage = Literal["bangla_script", "banglish", "english"]
AnswerLanguage = Literal["Bangla", "English"]

# The Bengali Unicode block. Bangla uses no other range, and nothing else in
# scope uses this one, so a single character in it settles the script question.
_BENGALI_START = "ঀ"
_BENGALI_END = "৿"

# The requirement in one table: Banglish is Bangla, and answers in Bengali
# script -- never transliterated back into Latin letters.
ANSWER_LANGUAGE: dict[QuestionLanguage, AnswerLanguage] = {
    "bangla_script": "Bangla",
    "banglish": "Bangla",
    "english": "English",
}

# Returned verbatim when retrieval finds nothing, so this text must never
# depend on a model call. That is the whole point of the no-documents path.
REFUSALS: dict[AnswerLanguage, str] = {
    "English": (
        "I do not have that information. Please contact the admission office."
    ),
    "Bangla": (
        "এই তথ্য আমার কাছে নেই। অনুগ্রহ করে ভর্তি অফিসের সাথে যোগাযোগ করুন।"
    ),
}

# High-frequency Banglish function words. Function words, not nouns, because
# Banglish borrows English nouns freely -- "CSE er tuition fee koto" is mostly
# English vocabulary held together by Bangla grammar, and the grammar is the
# part that identifies it.
BANGLISH_MARKERS = frozenset("""
er te ta ti ki ke koto kto kobe kokhon kothay kivabe kemon kmn ache achhe asche
hobe hoy hoye lagbe lage korte kore korbo parbo dite nite jonno jonne amar ami
amake apnar apni tumi tar nai nei bhorti vorti taka kichu kisu kono shob sob
bhalo valo chai chay jodi tahole abar onek aro kina porte pora shuru suru ekhon
dorkar holo ase hoise hole kharap bolen korar chaile
""".split())

# Counted against the above rather than used alone: some Banglish markers are
# also English words ("ache", "taka", "hole"), so one collision must not be
# able to flip an English sentence.
ENGLISH_MARKERS = frozenset("""
the is are was what how much many do does did i you we my your our a an of for
to in on at can could when where which who why need want tell me about there
offer with per and or if from have has get give it this that any all
""".split())


def classify_heuristic(text: str) -> QuestionLanguage | None:
    """Classify by script, then by function-word evidence.

    Returns None when neither vocabulary appears. That is deliberately not
    "english": a question the lexicon has no words for is exactly the case
    where a wrong guess reintroduces the bug, so the caller asks the model.
    """
    if any(_BENGALI_START <= char <= _BENGALI_END for char in text):
        return "bangla_script"

    words = re.findall(r"[a-z]+", text.lower())
    banglish = sum(word in BANGLISH_MARKERS for word in words)
    english = sum(word in ENGLISH_MARKERS for word in words)

    if banglish > english:
        return "banglish"
    if english > banglish:
        return "english"
    return None
