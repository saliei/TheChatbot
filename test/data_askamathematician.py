#!/usr/bin/env python3
# Scrapte the <askamathematician.com> for Q&A and save it in a json.

import re
import requests
from bs4 import BeautifulSoup

url = "https://www.askamathematician.com/"

# use a different user agent, the default agent is blocked
page = requests.get(url, headers={"User-Agent": "XY"})
print(page.status_code)
soup = BeautifulSoup(page.text, 'html.parser')
qs_links_tags = soup.find_all('a', text = re.compile("Q:*"))
qs_links = [atag["href"] for atag in qs_links_tags]

def get_post_body(url):
    qpage = requests.get(url, headers={"User-Agent": "XY"})
    qsoup = BeautifulSoup(qpage.text, 'html.parser')
    qtitle = qsoup.find("h1", {"class": "entry-title"}).text

    qdate = qsoup.find("span", {"class": "entry-date"}).text

    qcontent = ''
    content = qsoup.find("div", {"class": "entry-content"}).findAll('p')
    for p in content:
        qcontent += '\n' + ''.join(p.findAll(text=True))
    
    return qcontent
