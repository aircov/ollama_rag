# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 17:22
# @Author  : yaomw
# @Desc    : ik远程词典服务
import os
import time
from fastapi import FastAPI, Response, Body
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()


@app.api_route('/extdic', response_class=PlainTextResponse, methods=["GET", "HEAD"])
async def ext_dic(response: Response):
    file_name = 'ext.dic'

    # 文件不存在就创建
    if not os.access(file_name, os.F_OK):
        f = open(file_name, 'w')
        f.close()

    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()
    
    # 文件修改时间
    f_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(file_name).st_mtime))
    response.headers["Last-Modified"] = f_time
    response.headers["ETag"] = "2"
    return data


@app.api_route('/stopwords', response_class=PlainTextResponse, methods=["GET", "HEAD"])
async def stopwords(response: Response):
    file_name = 'stop.dic'
    # 文件不存在就创建
    if not os.access(file_name, os.F_OK):
        f = open(file_name, 'w')
        f.close()

    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()
    f_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(file_name).st_mtime))
    response.headers["Last-Modified"] = f_time
    response.headers["ETag"] = "2"
    return data

@app.post("/update_dict")
async def update_dict(type: str=Body(...,regex="ext|stopwords", title="字典类型", embed=True),
                      token: str=Body(..., title="token", embed=True)):
    type_dict = {
        'ext': 'ext.dic',
        'stopwords': 'stop.dic'
    }
    word_dict = type_dict[type]

    with open(word_dict, "a", encoding="utf-8") as f:
        f.write("\n"+token)

    return {"msg":"ok"}


if __name__ == "__main__":
    uvicorn.run(app="ik_remote_dict:app", reload=True, host="0.0.0.0", port=9527)


# import tornado.ioloop
# import tornado.web
# import os, time
#
# # 配置文件
# conf = {"port": 9527,
#         "ext_dic": "ext.dic",
#         "stopwords": "stop.dic"
#         }
#
#
# # Server句柄
# class MainHandler(tornado.web.RequestHandler):
#     # 初始化，传入字典文件
#     def initialize(self, file):
#         self.file = file
#         # 文件不存在就创建
#         if not os.access(self.file, os.F_OK):
#             f = open(self.file, 'w')
#             f.close()
#
#     # GET method
#     def get(self):
#         f = open(self.file, 'r', encoding='utf-8')
#         data = f.read()
#         f.close()
#         self.set_header("Content-Type", "text/plain; charset=UTF-8")
#         self.set_header("ETag", "2")
#         self.write(data)
#
#     # HEAD mothod
#     def head(self):
#         # 获取更新时间，设置为上次更改的标志
#         f_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(self.file).st_mtime))
#         self.set_header("Last-Modified", f_time)
#         self.set_header("ETag", "2")
#         self.set_header("Content-Length", "0")
#         self.finish()
#
#
# # 注册webMapping
# def make_app():
#     return tornado.web.Application([
#         (r"/extdic", MainHandler, {"file": conf["ext_dic"]}),
#         (r"/stopwords", MainHandler, {"file": conf["stopwords"]})
#     ])
#
#
# if __name__ == "__main__":
#     app = make_app()
#     app.listen(conf["port"])
#     tornado.ioloop.IOLoop.current().start()
