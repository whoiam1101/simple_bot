import sys
import asyncio

from pathlib import Path
from uuid import uuid4
from openai import OpenAI
from aiogram import Bot, types, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    telegram_bot_token: str
    openai_api_key: str


dp = Dispatcher()
config = Config.parse_file('.env')

bot = Bot(token=config.telegram_bot_token)

client = OpenAI(api_key=config.openai_api_key)
assistant = client.beta.assistants.create(
    name = "Question -> GPT -> Answer",
    instructions = "Question -> GPT -> Answer",
    tools = [{"type": "code_interpreter"}],
    model="gpt-3.5-turbo-1106"
)


async def get_audio(message: types.message) -> str:
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    file_name = f"audio/question{file_id}.mp3"
    await bot.download_file(file_path, file_name)
    return file_name


async def audio2text(file_name: str) -> str:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_name,
        response_format='text'
    )
    return transcript.text


async def get_answer(text: str) -> str:
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    client.beta.threads.runs.create_and_poll()
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value


async def text2audio(text: str) -> str:
    audio_file_name = f"audio/{str(uuid4())}.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
    )
    response.write_to_file(audio_file_name)
    return audio_file_name


@dp.message(Command("start"))
async def cmd_start(message: Message):
    audio_file_name = await get_audio(message)
    text = await audio2text(audio_file_name)
    Path(audio_file_name).unlink()  # remove file
    answer = await get_answer(text)
    audio_file_name = await text2audio(answer)
    audio_file = FSInputFile(audio_file_name)
    await message.reply_voice(audio_file)
    Path(audio_file_name).unlink()


async def main() -> None:
    print("BOT Start", file=sys.stderr)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("BOT Closed", file=sys.stderr)
