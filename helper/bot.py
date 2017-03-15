import telegram
import time
from tqdm import * #Progress Bar

def sendMessage(message: str):
    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendMessage(chat_id=chat_id, text=message)

def setProgressBar(i, rg):
    tqdm(range(rg))
