import requests
from bs4 import BeautifulSoup
import datetime
import json
rep = requests.get("https://careers.tencent.com/tencentcareer/api/post/Query?pageIndex=0&pageSize=10&language=zh-cn&area=cn")
jn = json.loads(rep.text)
import re
url = "https://careers.tencent.com/tencentcareer/api/post/Query?pageIndex=0"
print(re.search(r'pageIndex=(\d+)', url).group(0))

