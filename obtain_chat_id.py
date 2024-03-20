import telebot

API_KEY = '7074068104:AAHqQPwzW9bA1D1CfTIStsbqFxQOnuc6X08'
bot = telebot.TeleBot(API_KEY)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    chat_id = message.chat.id
    print("Chat ID:", chat_id)
    bot.reply_to(message, "This is your chat ID: " + str(chat_id))

bot.polling()
