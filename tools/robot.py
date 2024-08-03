"""
代码参考源：https://blog.csdn.net/qq_44809707/article/details/119959864

Robot测试通过 2022.11.21
"""

import socket


def send_msg(resp_dict):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    ip = '219.0.0.1'
    client.connect((ip, 5700))

    msg_type = resp_dict['msg_type']  # 回复类型（群聊/私聊）group/private
    number = resp_dict['number']  # 回复账号（群号/好友号）
    msg = resp_dict['msg']  # 要回复的消息

    # 将字符中的特殊字符进行url编码
    msg = msg.replace(" ", "%20")
    msg = msg.replace("\n", "%0a")

    if msg_type == 'group':
        payload = "GET /send_group_msg?group_id=" + str(
            number) + "&message=" + msg + " HTTP/1.1\r\nHost:" + ip + ":5700\r\nConnection: close\r\n\r\n"
    elif msg_type == 'private':
        payload = "GET /send_private_msg?user_id=" + str(
            number) + "&message=" + msg + " HTTP/1.1\r\nHost:" + ip + ":5700\r\nConnection: close\r\n\r\n"
    client.send(payload.encode("utf-8"))
    client.close()
    return 0


# 测试
if __name__ == '__main__':
    resp_dict = {'msg_type': 'group', 'number': 648887836, 'msg': '你好'}  # number对应Q号为私聊消息，对应群号为群消息
    # 测试发送 成功, 只要改msg即可
    send_msg(resp_dict)
