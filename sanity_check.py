import json

with open('adversarial.txt', 'r') as f:
    my_dict = json.loads(f.read())
    assert 'text' in my_dict
    text_list = my_dict['text']
    assert isinstance(text_list, list)
    assert len(text_list) == 1000
    for _text in text_list:
        assert isinstance(_text, str) and len(_text) >= 1
    print("\n\n----模拟输出检查完成----")
    print("----请上传镜像并提交----")
    print("----如本地验证通过，但线上报错，请联系言奇----")
