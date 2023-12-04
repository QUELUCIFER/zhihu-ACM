import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import jieba
import requests
import re
import json
import wordcloud
from PIL import Image            
from snownlp import SnowNLP
import jieba.posseg as seg
import pandas as pd



@st.cache_data
def read_deal_text():
 with open("zhihu_answers.txt", "r", encoding='utf-8') as f:
  txt = f.read()
 re_move = ["，", "。", " ", '\n', '\xa0', 'з」∠)_', '_(']
 # 去除无效数据
 for i in re_move:
  txt = txt.replace(i, " ")
 word = jieba.lcut(txt)  # 使用精确分词模式
 with open("wordcloud_zhihu_answers.txt", 'w') as file:
  for i in word:
   file.write(str(i) + ' ')
 #print("文本处理完成")


@st.cache_data
def GetZhihuText(question_id):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'cookie': '_zap=140285d3-c1fb-4859-adf8-41be6e3fdfeb; d_c0=AFAXtoOZKBePTowl16IU2e-kVQStglNP6uY=|1690641588; YD00517437729195%3AWM_TID=dkxdUvnoxqxFUEABQALQiYTkAH3U07CH; _xsrf=tH39ZQ3cnrRYFIRkMVq7GUJkkdJIWUih; __snaker__id=l2qKmPUndw1qM0xS; YD00517437729195%3AWM_NI=0l9qX%2BXo5ht3l84F6EXiZmPOfI8W9YXcP5X1rsz5eGISZIVnQbrn8RYFYrrrDUzklcRkE5mw%2FsicK5qeUD5IAockaZVHXfZ9h3sZVQHsbD1EamWklJ%2B4I%2FvFmMt2ODohcVM%3D; YD00517437729195%3AWM_NIKE=9ca17ae2e6ffcda170e2e6eeb3ed50948d8385ef5496868ab6c14f829a8eadd8339694bfaff77db2e9a29ab32af0fea7c3b92af4b4f792db219cb1fb89f739b0e8fca7b54f9cb7968ed345a697f8b8bc539cbda0d6db3bb2adb6d6e84d8b8d96aad56693bebe8fbb5a9b9a9793f17a888bb68cef6992b58bd6f07dedada784ca3fb3bb9ba5b84383b1ad8ce45296989887d933acedbdbbd55b8cadaea5b44fa3b3998cef42e9ec87d0f54eaf93bcd3ce50adb8adb7d037e2a3; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1701254740,1701261309,1701332197,1701345233; gdxidpyhxdE=yIc8VexrRRrG%2FAxfgLZbsHsa5UQRbmU10%5CiKsrYL04mW6%2Fov8Gbkjlf8ZCiS%2BCyeESUELy4rfON2f%2Bt%5CkhgdfjTYBvske56Im04tnGHDnqb09Gp4lbGaiqeA4s%5CYO4cq%5CzvKyay4%5CxR2dzzmR4SrxN6i5%5CTGg%5C6Wa1ex6krzBcuh7Ybk%3A1701349498815; captcha_session_v2=2|1:0|10:1701348755|18:captcha_session_v2|88:R3ZqV3htKzBBajhBT0JiMGlCQXdCWS9VajhBdityb3IrcGx3c0t0eGJPalI2c2JzMjZBdk5uTEh5M0RLL3VyUA==|7b304319b1f6548b68bb9cc16ced856d0c5bf24ca3f057b003f571763e734247; captcha_ticket_v2=2|1:0|10:1701348810|17:captcha_ticket_v2|728:eyJ2YWxpZGF0ZSI6IkNOMzFfZjIqU25RcUNDaUVxR2FUcTh0VmNaMHlpOS54RmZNbHRscmpiejZURk9iZzBqbmx2UWRUMXdFRUJNMUZTQmV4S1JfWVVmSG45bnhXVC44Uk1Uc1lmVkVkVXlMdEpJM0V0eHZlekFheXMqVEdyX3BCUWRSS1UyT0JBcXoxRHhDMngwWnNTYjZYd3NocjFTTVpsRlNqR1VlZDJkYWtJMUZpOHVHSHFXdEIwRVEuVVluUXpZdXJrUnJiUFllTnE1eWFtRllnLnA2YzAyQ2dWKmhJYTk2WHRSeWtIcnd5LmthQUhINFR0cFg1MTlGRWFfQyp6OGVRNkRxZUR0Z2Roa0ouYXFJSlFhTWZRaGtqeDlpTVUqNnFlRGN1c2lOM3ZoMUVpVkg5aVJNanljbjZuRHN0cGcxWGVzNWZlTjNlR2pYVXhYZUhBaTZzQWNFM3pWTTJJRHEwcnlHZ0JnaTBjZDBJY0hoTWRxbDVXaTloSGJYbloydzRiM01IWFJ3dFBjYjJ5cVVOSU5MQkNFR2dJZTFnOXZPVkdQUGo1dHNUMmNvZGdJcnFkdkc5UUYzdTFKKnVyVnpfNGlLNXF5QnFRRnNqdFJkWkFqLmIubkVpa3ZwcWF5Wk1JWHpFOGY2RzhpeFNSaWt6dGJ4VUNOUyp4Zy5ZRmtUSnhpUnFtTWtFeGY1T2JlQ00ySWc3N192X2lfMSJ9|4537f03761e1b2ff00c5d2beece345ca4c01a1ffc9b3b6e9a2001f7dbe40f9ff; z_c0=2|1:0|10:1701348890|4:z_c0|92:Mi4xbkVkNFRRQUFBQUFBVUJlMmc1a29GeVlBQUFCZ0FsVk5GdFJWWmdCZDVhZFlYeXJoVGRFR0dlbm0zUWRYb0RCc213|f9f27b45285bc13142a3878ef1a56b2b210c85faea5112bc266ff78f6ef5828b; KLBRSID=81978cf28cf03c58e07f705c156aa833|1701348934|1701345230; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1701348935'}

    with open(r'zhihu_answers.txt', 'a+', encoding='utf-8') as test:
        test.truncate(0)

    url = 'https://www.zhihu.com/question/' + question_id

    response = requests.get(url=url, headers=headers)

    title = re.findall('<title data-rh="true">(.*?) - 知乎</title>', response.text)[0]

    html_data = re.findall('<script id="js-initialData" type="text/json">(.*?)</script>', response.text)[0]

    # 下面要把字符串转为字典

    json_data = json.loads(html_data)
    json_dict = json_data['initialState']['entities']['answers']

    for i in json_dict.keys():
        content = json_dict[i]['excerpt']
        name = json_dict[i]['author']['name']
        with open('zhihu_answers' + '.txt', mode='a', encoding='utf-8') as f:
            f.write(f'{name},{content}\n')

@st.cache_data
def WordSeg():

    filename = 'zhihu_answers.txt'

    with open(filename, 'r', encoding='utf-8') as f:
        txt = f.read()
    words = jieba.lcut(txt)
    result = {}
    for word in words:
        if len(word) == 1:
            continue
        else:
            result[word] = result.get(word, 0) + 1

    lst = list(result.items())
    lst.sort(key=lambda x: -x[1])
    for i in range(20):
        word, count = lst[i]
        st.write("{0:<10}{1:>5}".format(word, count))

@st.cache_data
def Generate_WordCloud(src_img):

    read_deal_text()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    with open("wordcloud_zhihu_answers.txt", "r") as file:
     txt = file.read()

    mask = numpy.array(Image.open(src_img))      # 定义词频背景
    num = 15
    # 设置词云相关参数
    wc = wordcloud.WordCloud(
     #font_path = 'C:\Windows\Fonts\msyh.ttc', # 设置字体
     font_path = './msyh.ttc',
     background_color='white',                   # 背景颜色
     mask = mask,                                # 文字颜色+形状（有mask参数再设定宽高是无效的）
     max_words = num,                         # 显示词数
     max_font_size = 150 ,                        # 最大字号
     stopwords={'一个'},  # 设置停用词，不再词云图中表示
    )
    wc.generate(txt)
    wc.recolor(color_func=wordcloud.ImageColorGenerator(mask))
    plt.imshow(wc)
    plt.axis("off")
    #plt.show()
    st.pyplot()
    wc.to_file('result.jpg')
    with open("result.jpg", "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name="result.jpg",
            mime="image/png"
        )


@st.cache_data
def SentimentAnalysis():
    with open('zhihu_answers.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    analysis_list = list(jieba.cut(text))

    analysis_words = [(word.word, word.flag) for word in seg.cut(text)]

    keywords = [x for x in analysis_words if x[1] in ['a', 'd', 'v']]

    keywords = [x[0] for x in keywords]

    # Creating a variable called `pos_num` and assigning it the value of 0.
    pos_num = 0

    # Creating a variable called `neg_num` and assigning it the value of 0.
    neg_num = 0

    # This is a for loop that is looping through each word in the list of keywords.
    for word in keywords:
        # Creating a variable called `sl` and assigning it the value of the `SnowNLP` function.
        sl = SnowNLP(word)
        # This is an if statement that is checking to see if the sentiment of the word is greater than 0.5.
        if sl.sentiments > 0.5:
            # Adding 1 to the value of `pos_num`.
            pos_num = pos_num + 1
        else:
            # Adding 1 to the value of `neg_num`.
            neg_num = neg_num + 1
        # This is printing the word and the sentiment of the word.
        # print(word, str(sl.sentiments))

    # This is a string that is using the `format` method to insert the value of `pos_num` into the string.
    st.write('正面情绪关键词数量：{}'.format(pos_num))

    # This is a string that is using the `format` method to insert the value of `neg_num` into the string.
    st.write('负面情绪关键词数量：{}'.format(neg_num))

    # This is a string that is using the `format` method to insert the value of `pos_num` divided by the value of `pos_num`
    # plus the value of `neg_num` into the string.
    st.write('正面情绪所占比例：{}'.format(pos_num/(pos_num + neg_num)))

    sentiment_counts = [pos_num, neg_num]
    labels = ['Positive', 'Negative']
    colors = ['#66c2a5', '#fc8d62']
    explode = (.1,)
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sentiment_counts, colors=colors, autopct='%1.1f%%', startangle=90,
                                      labeldistance=0.9, pctdistance=1.1, textprops=dict(color="b"))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')
    plt.title("Sentiment Distribution")
    plt.setp(autotexts, size=10, weight="bold")
    #plt.show()
    st.pyplot()


# 设置网页标题
st.title('编程竞赛舆情分析')

# 展示一级标题
st.header('1. 选择你想要分析的竞赛')

data_dict = {
    '528320895':'如何评价2022年acm浙江省赛?',
    '455125989':'如何评价2021年ACM浙江省省赛？',
    '426055252':'如何评价2020浙江省ACM省赛？',
    '321950600':'如何评价2019浙江省ACM省赛？',
    '275281718':'如何评价2018浙江省ACM省赛？',
    '58860921':'如何评价2017浙江省ACM省赛？',
    '270765046':'如何评价电子科技大学 ACM 校赛高中生屠榜？',
    '593249699':'如何评价电子科技大学第21届ACM程序设计竞赛？'}

df = pd.DataFrame(list(data_dict.items()),columns=("question_id","zhihu_question"))

edited_df = st.data_editor(df,  num_rows = "dynamic")

success_for_word_segmentation = False
success_for_wordcloud = False
success_for_sentiment = False

# 用with进行Form的声明
with st.form(key='my_form'):
    question_id = st.text_input(label='输入你想分析的ACM相关编程竞赛对应的知乎问题的id')
    submit_button = st.form_submit_button(label='Submit')
    if len(question_id) != 8 and len(question_id) != 9:
        st.markdown("输入不合法！请重新输入！")
    else:
        st.write(f'你输入的问题id是{question_id}')
        GetZhihuText(question_id)
        success_for_word_segmentation = True


if success_for_word_segmentation:

    # 展示一级标题
    st.header('2. 爬取数据展示——词频统计')

    WordSeg()

    success_for_wordcloud = True

if success_for_wordcloud:
    # 展示一级标题
    st.header('3. 绘制词云图')

    src_img = st.file_uploader("Choose a local img")
    if src_img is not None:
        st.image(src_img)
        Generate_WordCloud(src_img)
        success_for_sentiment = True

if success_for_sentiment:
    # 展示一级标题
    st.header('4. 情感分析结果展示')
    SentimentAnalysis()












