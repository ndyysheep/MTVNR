import requests
import parsel
import random
import time

page_num = 0      # 记录页数
counter = 1845       # 记录爬取帖子数
for page in range(0, 11457351, 50):
    page_num += 1
    print(f'------------------正在爬取第{page_num}页数据------------------')# 上次爬到38页
    # 找到系统所在的链接地址（url)
    x = random.uniform(1, 4)
    time.sleep(x)
    url = f'https://tieba.baidu.com/f?ie=utf-8&kw=ps%E5%90%A7&fr={page}' ### 问题：不能翻页
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) Like Gecko'}
    # 发送网络请求代码 数据对比
    responce = requests.get(url=url, headers=headers)
    html_data = responce.text
    # print(html_data)

    # 数据解析
    # 3.1转换数据类型
    selector = parsel.Selector(html_data)
    # print(selector)
    # 3.2解析数据
    title_url = selector.xpath('//div[@class="threadlist_lz clearfix"]/div/a/@href').getall()
    # print(title_url)
    for li in title_url:
        # 拼接完整的帖子链接
        all_url = 'https://tieba.baidu.com/' + li
        print('当前帖子链接为：', all_url)

        # 继续发送每一个帖子的链接请求
        x = random.uniform(1, 3)
        time.sleep(x)
        responce_2 = requests.get(url=all_url, headers=headers).text

        # 第二次解析 解析图片地址
        responce_2_selector = parsel.Selector(responce_2)
        result_list = responce_2_selector.xpath('//cc/div/img[@class="BDE_Image"]/@src').getall()

        # 发送图片的链接请求
        a_counter = 1
        for result in result_list:
            x = random.uniform(0, 2)
            time.sleep(x)
            # print("当前帖子图片链接：", result)
            img_data = requests.get(url=result, headers=headers).content  # 图片数据
            # 4.保存数据

            # 准备文件名-
            file_name = result.split('/')[-1]
            file_name = file_name.split('?')[0]
            with open('IMG\\' + str(counter)+"."+str(a_counter)+".jpg", mode='wb')as f:
                f.write(img_data)
                print("保存完成：", str(counter)+"."+str(a_counter))
            a_counter += 1
        counter += 1
