import requests
from bs4 import BeautifulSoup
import time

'''
tag_span = soup.find_all("span" , class_ = "sc-6oxm01-2 hiTIMq")
allList = []
for tag in tag_span:
    print(tag)  
    allList.append(tag.text)
    print(tag.text)
print(allList[2::3])
'''


'''
tag_divs = soup.find_all("a" , class_ = "sc-1v1d5rx-3 kPUUNB")
# print(tag_divs)

for tag in tag_divs:
    # print(tag)
    if tag.find('span'): # 是否有<span>標籤
        tag_span = tag.find("span")
        print(tag_span.text)
for tag in tag_divs:
    print("https://www.dcard.tw" + tag['href'])
'''

########################
# 正確版
########################
"""
url = "https://www.dcard.tw/f"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
           'AppleWebKit/537.36 (KHTML, like Gecko)'
           'Chrome/63.0.3239.132 Safari/537.36'}
r = requests.get(url , headers = headers , cookies = {"over18": "1"})
r.encoding = "utf8"
soup = BeautifulSoup(r.text , "lxml")


i = 0
j = 1
k = 2

tag_divs = soup.find_all("a" , class_ = "sc-1v1d5rx-3 kPUUNB")
tag_span = soup.find_all("span" , class_ = "sc-6oxm01-2 hiTIMq")
allList = []

for tag in tag_span:    
    allList.append(tag.text)

for tag in tag_divs:
    if tag.find("span"):
        tag_title = tag.find("span")
        print(f"標題：{tag_title.text}")
        print(f"網址：{'https://www.dcard.tw' + tag['href']}")
        while i < len(allList):
            print(f"板名：{allList[i]}")
            print(f"學校：{allList[j]}")
            print(f"時間：{allList[k]}")
            i += 3
            j += 3
            k += 3
            break
        print("===---===---===---===---===")
"""


########################
# 正確版
########################
url = "https://www.dcard.tw/f"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            'AppleWebKit/537.36 (KHTML, like Gecko)'
            'Chrome/63.0.3239.132 Safari/537.36'}
r = requests.get(url , headers = headers , cookies = {"over18": "1"})

if r.status_code == requests.codes.ok:
    r.encoding = "utf8"
    soup = BeautifulSoup(r.text , "lxml")


    i = 0
    j = 1
    k = 2

    tag_divs = soup.find_all("a" , class_ = "sc-1v1d5rx-3 kPUUNB")
    tag_span = soup.find_all("span" , class_ = "sc-6oxm01-2 hiTIMq")
    allList = []

    for tag in tag_span:    
        allList.append(tag.text)

    for tag in tag_divs:
        if tag.find("span"):
            tag_title = tag.find("span")
            print(f"標題：{tag_title.text}")
            print(f"網址：{'https://www.dcard.tw' + tag['href']}")
            while i < len(allList):
                print(f"板名：{allList[i]}")
                print(f"學校：{allList[j]}")
                print(f"時間：{allList[k]}")
                i += 3
                j += 3
                k += 3
                break
            print("===---===---===---===---===")
    time.sleep(2)
else:
    print("HTTP請求錯誤..." + url)

