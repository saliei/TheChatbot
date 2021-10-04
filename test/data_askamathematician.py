#!/usr/bin/env python3
# Scrapte the <askamathematician.com> for Q&A and save it in a json.

import re
import requests
from bs4 import BeautifulSoup

url = "https://www.askamathematician.com/"

counter = 0

# while True:
    # r = requests.get(url)
    # file_like_obj = io.StringIO(r.text) #Turns the requested output into a file like objet
    # lines = file_like_obj.read()
    # for l in lines:
        # counter += 1 #Update the counter from proper filenames
        # soup = BeautifulSoup(lines)
        # findID = re.findall(r'post-(.*)\' itemprop', lines) #This retrieves each post's unique ID-number for the text.
        # print(findID[0]) #for debugging
        # div = soup.find(id="post-body-" + findID[0]) #This retrieves each post content
        # print(div)
        # with open(str(counter) + ".html", "w") as outputfile: #open file
            # outputfile.write(str(div)) #write to file
            # matchObj = re.findall(r'blog-pager-older-link\' href=\'(.*)\' id', lines) # This extract the "Older" post
            # next_url = matchObj[0]
            # print("Next URL for scraping: " + next_url)
            # print("Press CTRL-C to exit the program.")
            # url = next_url # This changes the variable in the beginning of the script

# use a different user agent, the default agent is blocked
page = requests.get(url, headers={"User-Agent": "XY"})
print(page.status_code)
soup = BeautifulSoup(page.text, 'html.parser')
qs_links_tags = soup.find_all('a', text = re.compile("Q:*"))
qs_links = [atag["href"] for atag in qs_links_tags]

# test for one of the questions
# qpage = requests.get(qs_links[0], headers={"User-Agent": "XY"})
# qsoup = BeautifulSoup(qpage.text, 'html.parser')
# q = qsoup.find("h1", {"class": "entry-title"})
# q = q.text

def get_post_body(url):
    qpage = requests.get(url, headers={"User-Agent": "XY"})
    qsoup = BeautifulSoup(qpage.text, 'html.parser')
    qtitle = qsoup.find("h1", {"class": "entry-title"})
    qtitle = qtitle.text

    qcontent = ''
    content = qsoup.find("div", {"class": "entry-content"}).findAll('p')
    for p in content:
        qcontent += '\n' + ''.join(p.findAll(text=True))
    
    return qcontent
