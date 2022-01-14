from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import numpy as np


IMG_SIZE = 300
CHANNELS = 3
PHOTOS_PATH = r'C:\Users\lubab\Desktop\TGbot'
CLASSES = ['экшeн', 'приключение', 'анимация', 'биографический', 'комедия', 'криминал', 'документальный', 'драма',
           'семейный',
           'фэнтези', 'историческое', 'хоррор', 'музыкальный', 'мьюзикл', 'мистика', 'новостной', 'реалити шоу',
           'романтика',
           'фантастика', 'короткометражка', 'спорт', 'триллер', 'военный', 'вестерн'
           ]


def on_start(update, context):
    chat = update.effective_chat
    context.bot.send_message(chat_id=chat.id, text="Привет, никогда не задумывался, какой жанр у твоей жизни? \
                        \nПрисылай фотографию, узнаешь =)")


def read_img(path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE, CHANNELS))
    img = image.img_to_array(img)
    img = img / 255.0
    return img


def get_path():
    files = os.listdir(PHOTOS_PATH)
    files = [os.path.join(PHOTOS_PATH, file) for file in files]
    files = [file for file in files if os.path.isfile(file)]
    return max(files, key=os.path.getctime)


def genres_def(file_path):
    img = read_img(file_path)
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, CHANNELS)
    y_pred = model.predict(img)
    top = np.argsort(y_pred[0])[:-4:-1]
    answer = [CLASSES[x] for x in top]
    return ', '.join(answer)


def on_message(update, context):
    chat = update.effective_chat
    context.bot.get_file(update.message.photo[-1]).download()

    try:
        file_path = get_path()
        output = genres_def(file_path)
        context.bot.send_message(chat_id=chat.id,
                                 text="Ого, если бы по твоей жизни снимали кино, то оно было бы жанров: " + output)
        os.remove(file_path)
    except:
        context.bot.send_message(chat_id=chat.id, text="Упс. Что-то пошло не так. Попробуй снова")


if __name__ == '__main__':
    model = tf.keras.models.load_model('genres_definition.h5')
    TOKEN = "TOKEN"
    updater = Updater(TOKEN, use_context=True)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", on_start))
    dispatcher.add_handler(MessageHandler(Filters.all, on_message))

    updater.start_polling()
    updater.idle()
