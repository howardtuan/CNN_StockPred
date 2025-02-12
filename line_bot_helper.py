from linebot import LineBotApi
from linebot.models import TextSendMessage

# Line Bot 配置
LINE_ACCESS_TOKEN = 'YOUR_LINE_ACCESS_TOKEN'
GROUP_ID = 'YOUR_GROUP_ID' 


# 初始化 Line API
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)

def send_message_to_group(text, accuracy=None):
    """
    發送文字訊息到群組

    :param text: 要傳送的主要訊息
    :param accuracy: 模型準確率，可選
    """
    pass
    # if accuracy is not None:
    #     text += f"\n準確率：{accuracy}"
    # message = TextSendMessage(text=text)
    # line_bot_api.push_message(GROUP_ID, message)
