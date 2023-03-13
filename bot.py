from model import RNNNet, load_model, predict_sentiment
from tg_settings import TG_TOKEN
from preprocessing import data_preprocessing, preprocess_single_string, padding, get_text_review
from aiogram import Bot, Dispatcher, executor, types
import re
import string
import logging
import json



API_TOKEN = TG_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

url = 'https://www.imdb.com/?ref_=nv_home'
link_text = "IMDB.com"
hyperlink = f'<a href="{url}">{link_text}</a>' 

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.username
    user_id = message.from_user.id
    text = f'''Hello ,{user_name}, this bot was created to receive feedback about the film and analyze the emotional coloring of the film review. 
Just send me the text of the review or link to review on {link_text} and I will be able to answer whether it is: "Positive" or "Negative"'''
    logging.info(f'{user_name=}{user_id=} sent message:{message.text}')
    await message.reply(text)

@dp.message_handler()
async def toxic_level(message: types.Message):
    user_name = message.from_user.username
    user_id = message.from_user.id
    if message.text.startswith('https://www.imdb.com'):
        text = get_text_review(message.text)
    else:
        text = message.text
    result = f''' This film review is more likely:{predict_sentiment(text)} 
Text of review: {text}'''
    logging.info(f'{user_name=}, {user_id=} sent message: {text=} bot_return: {result=}')
    await bot.send_message(user_id, result)


if __name__ == '__main__':
    executor.start_polling(dp)    



