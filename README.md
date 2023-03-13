# @imdb_sent_bot: 

This bot was created to receive feedback about the film and analyze the emotional coloring of the film review. 
Just send me the text of the review or link to review on [Imdb.com](https://www.imdb.com/) and I will be able to answer whether it is: "Positive" or "Negative"

## How to run the project:

**For the bot to work properly, you need to create a virtual environment using requirements.txt**, for this, write in the terminal:

```
git clone https://github.com/trojanof/toxic_comments_analisys_tg_bot #make a clone of this repository on your computer
cd toxic_comments_analisys_tg_bot #go to the repository folder
python -m venv .venv # create a new virtual environment in the project folder
pip install -r requirements.txt #install the required libraries in this environment
```
Create a new telegram bot for yourself using the [BotFather](https://telegram.me/BotFather) bot and save the received token to a file
tg_settings.py

To start your bot, run the command in the terminal:
```
python bot.py