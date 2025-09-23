import re

import langdetect
from langdetect import detect

from app.be_core.logger import logger
from app.utils.openai_utils import chat_completion_with_retry

# Language to voice mapping
LANGUAGE_VOICE_MAP = {
    "en": "en-US-BrianMultilingualNeural",  # Default English voice
    "hi": "hi-IN-KavyaNeural",  # Hindi voice
    "gu": "gu-IN-DhwaniNeural",  # Gujarati voice
}


# Function to detect language
def detect_language(segment):
    return detect(segment)  # Returns language code (e.g., 'en', 'hi')


async def convert_to_human_friendly_voice_text(text: str) -> str:
    """Convert raw text to human-friendly voice text using GPT-4o."""
    if not text or len(text.strip()) == 0:
        return text

    system_prompt = """You are a text-to-speech assistant. Your job is to convert raw text into human-friendly voice text that sounds natural when spoken aloud.

Guidelines:
1. Remove or convert special formatting characters (#, *, _, `, [], {}, (), <>, |, etc.) to natural speech
2. Convert bullet points (•, ◦, ▪, etc.) to natural list format
3. Replace URLs with "website link" or "website"
4. Convert abbreviations and acronyms (like ICC, ODI, ESPN) to their spoken form (I C C, O D I, E S P N)
5. Handle table-like structures and formatting to make them sound natural
6. Preserve all important information while making it conversational
7. Keep the same language and meaning
8. Make it sound like natural speech, not like reading formatted text

Examples:
- "• Full Name : Virat Kohli" → "Full Name is Virat Kohli"
- "ICC Test Player of the Year" → "I C C Test Player of the Year"
- "https://www.imdb.com/name/nm8667438/bio" → "website link"
- "Named ICC Cricketer of the Year at the **ICC** annual awards" → "Named I C C Cricketer of the Year at the I C C annual awards"

Return only the converted text, nothing else."""

    user_prompt = f"Convert this text to human-friendly voice text:\n\n{text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = chat_completion_with_retry(
            messages=messages,
            temperature=0.1,  # Low temperature for consistent formatting
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Error converting text to human-friendly voice text: {str(e)}")
        # Return original text if conversion fails
        return text


# Function to construct SSML with language-specific voices
def construct_ssml_multilingual(text):
    ssml = '<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="en-US">'
    if not text:
        ssml += "</speak>"
        return ssml

    segments = re.split(r"[?!]", text)
    for segment in segments:
        if segment:
            try:
                lang_code = detect_language(segment)
                voice_name = LANGUAGE_VOICE_MAP.get(
                    lang_code, "en-US-BrianMultilingualNeural"
                )  # Default to English
            except langdetect.lang_detect_exception.LangDetectException as e:
                logger.debug(f"Segment: {segment} can't processed by lang_detect")
                logger.error(f"LangDetectException : {str(e)}")
                voice_name = "en-US-BrianMultilingualNeural"
            ssml += f'<voice name="{voice_name}">{segment.strip()}.</voice> '

    ssml += "</speak>"
    return ssml


def construct_ssml_english(text: str) -> str:
    ssml = '<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="en-US">'
    if text:
        ssml += f'<voice name="en-US-BrianMultilingualNeural">{text.strip()}</voice>'
    ssml += "</speak>"
    return ssml


# Async versions that use GPT-4o for human-friendly voice text conversion
async def construct_ssml_multilingual_with_gpt(text: str) -> str:
    """Construct SSML with GPT-4o human-friendly voice text conversion."""
    if not text:
        return '<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="en-US"></speak>'

    # Convert text to human-friendly voice text using GPT-4o
    human_friendly_text = await convert_to_human_friendly_voice_text(text)

    # Use the converted text for SSML construction
    return construct_ssml_multilingual(human_friendly_text)


async def construct_ssml_english_with_gpt(text: str) -> str:
    """Construct English SSML with GPT-4o human-friendly voice text conversion."""
    if not text:
        return '<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="en-US"></speak>'

    # Convert text to human-friendly voice text using GPT-4o
    human_friendly_text = await convert_to_human_friendly_voice_text(text)

    # Use the converted text for SSML construction
    return construct_ssml_english(human_friendly_text)


if __name__ == "__main__":
    # Example multilingual text
    input_text = "Hello, how are you? नमस्ते, आप कैसे हैं? Have a great day!"

    # Generate SSML
    ssml_gen = construct_ssml_multilingual(input_text)
    print("Generated SSML:\n", ssml_gen)

    print("Generated SSML:\n", construct_ssml_english("Hello, how are you?"))
