#
# 人脸检测和属性分析 WebAPI 接口调用示例
# 运行前：请先填写Appid、APIKey、APISecret以及图片路径
# 运行方法：直接运行 main 即可 
# 结果： 控制台输出结果信息
# 
# 接口文档（必看）：https://www.xfyun.cn/doc/face/xf-face-detect/API.html
#

from datetime import datetime
from wsgiref.handlers import format_date_time
from time import mktime
import hashlib
import base64
import hmac
from urllib.parse import urlencode
import os
import traceback
import pandas as pd
import json
import requests


class AssembleHeaderException(Exception):
    def __init__(self, msg):
        self.message = msg


class Url:
    def __init__(self, host, path, schema):
        self.host = host
        self.path = path
        self.schema = schema
        pass


# 进行sha256加密和base64编码
def sha256base64(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    digest = base64.b64encode(sha256.digest()).decode(encoding='utf-8')
    return digest


def parse_url(requset_url):
    stidx = requset_url.index("://")
    host = requset_url[stidx + 3:]
    schema = requset_url[:stidx + 3]
    edidx = host.index("/")
    if edidx <= 0:
        raise AssembleHeaderException("invalid request url:" + requset_url)
    path = host[edidx:]
    host = host[:edidx]
    u = Url(host, path, schema)
    return u


def assemble_ws_auth_url(requset_url, method="GET", api_key="", api_secret=""):
    u = parse_url(requset_url)
    host = u.host
    path = u.path
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    # print(date)
    # date = "Thu, 12 Dec 2019 01:57:27 GMT"
    signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(host, date, method, path)
    # print(signature_origin)
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
        api_key, "hmac-sha256", "host date request-line", signature_sha)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    # print(authorization_origin)
    values = {
        "host": host,
        "date": date,
        "authorization": authorization
    }

    return requset_url + "?" + urlencode(values)


def gen_body(appid, img_path, server_id):
    with open(img_path, 'rb') as f:
        img_data = f.read()
    body = {
        "header": {
            "app_id": appid,
            "status": 3
        },
        "parameter": {
            server_id: {
                "service_kind": "face_detect",
                # "detect_points": "1", #检测特征点
                "detect_property": "1",  # 检测人脸属性
                "face_detect_result": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "json"
                }
            }
        },
        "payload": {
            "input1": {
                "encoding": "jpg",
                "status": 3,
                "image": str(base64.b64encode(img_data), 'utf-8')
            }
        }
    }
    return json.dumps(body)


def run(appid, apikey, apisecret, img_path, server_id='s67c9c78c'):
    url = 'http://api.xf-yun.com/v1/private/{}'.format(server_id)
    request_url = assemble_ws_auth_url(url, "POST", apikey, apisecret)
    headers = {'content-type': "application/json", 'host': 'api.xf-yun.com', 'app_id': appid}
    # print(request_url)
    response = requests.post(request_url, data=gen_body(appid, img_path, server_id), headers=headers)
    resp_data = json.loads(response.content.decode('utf-8'))
    # print(resp_data)
    if not resp_data.get('payload'):
        return None
    return base64.b64decode(resp_data['payload']['face_detect_result']['text']).decode()


def run_once(path):
    text = json.loads(run(
        appid='a4f4c658',
        apisecret='NGZkMWQzODUyMzJkNzRmZjMyZTAzZGU0',
        apikey='28ab5c28403ce4501a87c72bb28057e4',
        img_path=path
    ))
    if not text or text['ret'] != 0 or not text.get('face_1'):
        print(text['ret'])
        return None
    return text['face_1']['property']


# 请填写控制台获取的APPID、APISecret、APIKey以及要检测的图片路径
if __name__ == '__main__':
    dir_ = "../img_align_celeba"
    img_list = sorted(os.listdir(dir_))
    d = dict({'name': [], 'gender': [], 'glass': [], 'beard': [], 'hair': [], 'mask': [], 'expression': []})
    keys = list(d.keys())
    df = pd.DataFrame.from_dict(d)
    filename = '../partition/label.txt'
    # df.to_csv(filename, header=True, index=False, columns=keys, mode='w')
    # 199, 1707, 1864, 1884, 1919, 2407, 2432, 2548, 2657, 3347, 5590, 5735, 5842, 7132, 13043
    for i, img in enumerate(img_list):
        if i <= 96398:
            continue
        print("{}/{}: {} ".format(i, len(img_list), img), end="")
        try:
            res = run_once(dir_ + '/' + img)
        except TypeError:
            continue
        except ConnectionResetError:
            continue
        if not res:
            continue
        print("success!")
        res['name'] = img
        df = pd.DataFrame(res, index=[0])
        df.to_csv(filename, header=False, index=False, columns=keys, mode='a')
