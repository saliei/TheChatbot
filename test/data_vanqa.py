#!/usr/bin/env python3
# Scrape the <https://van.physics.illinois.edu/qa/> website for Q&A posts.

import re
import requests
from bs4 import BeautifulSoup

# sample catgory page for getting all the categories
url = "https://van.physics.illinois.edu/qa/subcategory.php?sub=Fire"

page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
# find all subcategories
subcats = soup.find_all("a", attrs = {"href": re.compile("subcategory.php*")})
url_prefix = "https://van.physics.illinois.edu/qa/"
subcat_links = [url_prefix + subcat["href"] for subcat in subcats]

# find all Q&A links
ques_links = []
for subcat_link in subcat_links:
    subcat_page = requests.get(subcat_link)
    subcat_soup = BeautifulSoup(subcat_page.text, 'html.parser')
    subcat_quns = subcat_soup.find_all("a", attrs = {"href": re.compile("listing.php*")})
    subcat_quns_links = [url_prefix + subcat_quns_link["href"] for subcat_quns_link in subcat_quns]
    # TODO: save links so we don't have to run finding links each time
    for subcat_quns_link in subcat_quns_links:
        print(subcat_quns_link)
        ques_links.append(subcat_quns_link)
        print("Total Q&A posts: {}".format(len(ques_links)))
    
# example Q&A page
url_qa = "https://van.physics.illinois.edu/qa/listing.php?id=2353&t=magnetic-field-strength-at-the-end-of-a-solenoid"
def get_qa(url_qa):
    page_qa = reuqests.get(url_qa)
    soup_qa = BeautifulSoup(page_qa.text, 'html.parser')
    content_qa = {"url": url_qa}
    # find all Q&A sections
    qas = soup_qa.find_all("div", {"class": "listingletter"})
    qas_text = []
    for sec in qas:
        if sec.string == "Q:":
            ques = sec.next_sibling.text
            qas_text.append(ques)
        if sec.string == "A:":
            answ = sec.next_sibling.text
            qas_text.append(answ)

    counter = 0
    for i in range(len(qas_text)):
        if i % 2 == 0:
            counter += 1
            content_qa["Q&A{}".format(counter)] = [qas_text[i], qas_text[i+1]]

    return content_qa

content = get_qa(url_qa)

