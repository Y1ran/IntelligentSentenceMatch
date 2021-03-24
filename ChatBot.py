# -*- coding:utf-8 -*-
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
import requests
import os

class UNIT:
    def __init__(self, api_key, api_secret):
        self.access_token = None
        self.url = None

        self.set_access_token(api_key, api_secret)

    def set_access_token(self, api_key, api_secret):
        host = 'https://aip.baidubce.com/oauth/2.0/token?' \
               'grant_type=client_credentials&' \
               'client_id={0}&' \
               'client_secret={1}'.format(api_key, api_secret)
        response = requests.post(host)
        if response:
            self.access_token = response.json()['access_token']
            print("access_token响应请求成功，请输入对话：")
        else:
            print("access_token响应请求失败，请排查原因")

    def query(self, query_text, app_id):
        self.url = 'https://aip.baidubce.com/rpc/2.0/unit/service/chat?access_token=' + self.access_token
        post_data = """{
                    "bot_session": "",
                    "log_id": "7758521",
                    "service_id": "%s",	# 你的聊天机器人的ID
                    "request": {
                        "bernard_level": 1,
                        "client_session": "{\\\"client_results\\\":\\\"\\\", \\\"candidate_options\\\": []}",
                        "query": "%s",	# 要查询的文字
                        "query_info": {
                            "asr_candidates": [],
                            "source": "KEYBOARD",
                            "type": "TEXT"
                        },
                        "updates": "",
                        "user_id": "8888"
                    },
                    "session_id": "",
                    "version": "2.0"
                }""" % (app_id, query_text)
        post_data = post_data.encode('utf-8')
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(self.url, data=post_data, headers=headers)
        if response:
            return response.json()['result']['response_list'][0]['action_list'][0]['say']

if __name__ == "__main__":
    print("正在接入闲聊模式，请等待AI上线...")
    app_id = "23815184"
    # 1086918
    # 23815184
    inputs = sys.argv[1]

    api_key = "SVy07kGrs4h7zCtkVrPOB6wZ"
    api_secret = "TpFxIA0hmLcDqlYNuXjZfBqGi9HxaVvk"
    chatBot = UNIT(api_key, api_secret)
