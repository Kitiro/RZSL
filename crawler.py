#!/usr/bin/env python3
# coding=utf-8
'''
Author: Kitiro
Date: 2021-10-28 15:18:43
LastEditTime: 2021-11-21 15:20:52
LastEditors: Kitiro
Description: 
FilePath: /zzc/exp/web_zsl/bing_crawler.py
'''
import requests
from lxml import etree
import re
import time
import os 
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from tqdm import tqdm
    
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
}


NUM = 1000 # 1000 downloaded images for each class
# class Bing():
#     def __init__(self, max_num=1000) -> None:
#         self.max = max_num
#     def save_img(self, url, save_path, index):
#         # img_name = url[-10:]
#         # name = re.sub('/', '', img_name)  # img_name中出现/，将其置换成空
#         try:
#             res = requests.get(url, headers=headers)
#         except OSError:
#             print('出现错误，错误的url是:', url)
#             return 0
#         else:
#             with open(os.path.join(save_path,str(index)+'.jpg'), 'wb')as f:
#                 try:
#                     f.write(res.content)
#                     return 1
#                 except OSError:
#                     print('无法保存，url是：', url)
#                     return 0


#     # 获取全部图片url:原图
#     def parse_source_img(self, img_list, url):
#         response = requests.get(url, headers=headers)
#         response.encoding = response.apparent_encoding
#         data = response.content.decode('utf-8', 'ignore')
#         html = etree.HTML(data)
#         conda_list = html.xpath('//a[@class="iusc"]/@m')
#         for i in conda_list:
#             img_url = re.search('"murl":"(.*?)"', i).group(1)
#             if img_url not in img_list:
#                 img_list.append(img_url)
#         return img_list[:self.max]
    

class Bing():
    def __init__(self, max_num=1000):
        self.max = max_num
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def save_img(self, url, save_path, index):
        try:
            res = requests.get(url, headers=headers)
        except OSError:
            print('出现错误，错误的url是:', url)
            return 0
        else:
            with open(os.path.join(save_path,str(index)+'.jpg'), 'wb')as f:
                try:
                    f.write(res.content)
                    return 1
                except OSError:
                    print('无法保存，url是：', url)
                    return 0

    # 获取全部图片url缩略图
    def parse_img(self, img_list, url):
        self.driver.get(url)
        time.sleep(2)
        # 模拟滚到到底部
        height = 1000
        js = 'window.scrollBy(0,%i)' % height
        for _ in range(100):
            self.driver.execute_script(js)
            time.sleep(0.1)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        img_urls = soup.find_all(name='img',attrs={"class":"mimg"})#按照字典的形式给attrs参数赋值
        for u in img_urls:
            uu = re.search(r'https://(.*)\?', str(u))
            if uu==None:
                continue
            img_url = uu.group(0).replace('?','')
            if img_url not in img_list:
                img_list.append(img_url)   # 用https匹配img地址
        return img_list[:self.max]


class Baidu():
    def __init__(self, max_num=1000):
        self.max = max_num
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def save_img(self, url, save_path, index):
        try:
            res = requests.get(url, headers=headers)
        except OSError:
            print('出现错误，错误的url是:', url)
            return 0
        else:
            with open(os.path.join(save_path,str(index)+'.jpg'), 'wb')as f:
                try:
                    f.write(res.content)
                    return 1
                except OSError:
                    print('无法保存，url是：', url)
                    return 0

    # 获取全部图片url缩略图
    def parse_img(self, img_list, url):
        self.driver.get(url)
        time.sleep(2)
        # 模拟滚到到底部
        height = 1000
        js = 'window.scrollBy(0,%i)' % height
        for _ in range(100):
            self.driver.execute_script(js)
            time.sleep(0.1)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        img_urls = soup.find_all(name='img',attrs={"class":"mimg"})#按照字典的形式给attrs参数赋值
        for u in img_urls:
            uu = re.search(r'https://(.*)\?', str(u))
            if uu==None:
                continue
            img_url = uu.group(0).replace('?','')
            if img_url not in img_list:
                img_list.append(img_url)   # 用https匹配img地址
        return img_list[:self.max]

# 读取需要搜索的class list
def get_search_list(name):
    if name == 'CUB':
        fp = 'data/'+name+'_data/cub_class.txt'
        class_list = [n.strip() for n in open(fp).readlines()]
        # search_list = [n.replace('_', ' ') for n in class_list]
    else:
        fp = 'data/'+name+'_data/trainvalclasses.txt'
        class_list1 = [n.strip() for n in open(fp).readlines()]
        fp = 'data/'+name+'_data/testclasses.txt'
        class_list2 = [n.strip() for n in open(fp).readlines()]
        class_list = class_list1 + class_list2
        if name == 'AWA2':
            class_list = [n.replace('+', ' ') for n in class_list]
    print('DataSet:', name, 'Contains', str(len(class_list))+' classes.')
    print(class_list[:20])
    return class_list


# 主函数
def main():
    dataset = ['AWA2', 'CUB', 'SUN', 'APY', 'FLO']
    save_dir = 'web_data_baidu'
    
    for ds in dataset[4:]:
        # original_name_list, search_name_list
        class_list = get_search_list(ds)

        # crawler = Bing(1000)
        crawler = Baidu(1000)
        for i in range(len(class_list)):
            save_path = os.path.join(save_dir, ds, class_list[i])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                continue
            img_list = []
            search_name = class_list[i]
            # bing
            # url = 'https://cn.bing.com/images/search?q={}&form=BESBTB&first=1&tsc=ImageBasicHover&ensearch=1'.format(search_name)
            https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E7%BE%9A%E7%BE%8A&oq=%E7%BE%9A%E7%BE%8A&rsp=-1
            crawler.parse_img(img_list, url)
            print(str(i)+'/'+str(len(class_list)), 'For class', search_name)
            print('Obtain', len(img_list), 'image urls')
            success = 0
            for idx, img_url in enumerate(tqdm(img_list)):
                res = crawler.save_img(img_url, save_path, idx)
                success += res 
                time.sleep(0.1)
            print(success, 'imgs saved.')
            print('-'*20)

if __name__ == '__main__':
    main()