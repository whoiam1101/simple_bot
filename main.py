import asyncio
import whisper

from uuid import uuid4
from openai import OpenAI
from aiogram import Bot, types, Dispatcher
from aiogram.types import Message, ContentType
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    telegram_bot_token: str
    openai_api_key: str


dp = Dispatcher()
config = Config.parse_file('.env')

bot = Bot(token=config.telegram_bot_token)

audio2text_model = whisper.load_model("base")

client = OpenAI(api_key=config.openai_api_key)
assistant = client.beta.assistants.create(
    name = "Question -> GPT -> Answer",
    instructions = "Question -> GPT -> Answer",
    tools = [{"type": "code_interpreter"}],
    model="gpt-3.5-turbo-1106"
)


@dp.message_handler(content_types=[ContentType.VOICE])
async def get_audio(message: types.message) -> str:
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    file_name = f"audio/question{file_id}.mp3"
    await bot.download_file(file_path, file_name)
    return file_name


def audio2text(file_name: str) -> str:
    audio = whisper.load_audio(file_name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(audio2text_model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(audio2text_model, mel, options)
    return result.text


def get_answer(text: str) -> str:
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages[0].content[0].text.value


def text2audio(text: str) -> str:
    audio_file_name = f"audio/{str(uuid4())}.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
    )
    response.write_to_file(audio_file_name)
    return audio_file_name


@dp.message()
async def cmd_start(message: Message):
    audio_file_name = get_audio(message)
    text = audio2text(audio_file_name)
    answer = get_answer(text)
    audio_file_name = text2audio(answer)
    message.answer(answer)
    message.answer_audio(audio_file_name)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("BOT Closed")
