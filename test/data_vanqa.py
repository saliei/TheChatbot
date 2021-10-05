#!/usr/bin/env python3
# Scrape the <https://van.physics.illinois.edu/qa/> website for Q&A posts.

import re
import requests
from bs4 import BeautifulSoup

# sample catgory page for getting all the categories
url = "https://van.physics.illinois.edu/qa/subcategory.php?sub=Fire"
# example Q&A page
url_qa = "https://van.physics.illinois.edu/qa/listing.php?id=2353&t=magnetic-field-strength-at-the-end-of-a-solenoid"

def get_qa_links(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    # find all subcategories
    subcats = soup.find_all("a", attrs = {"href": re.compile("subcategory.php*")})
    url_prefix = "https://van.physics.illinois.edu/qa/"
    subcat_links = [url_prefix + subcat["href"] for subcat in subcats]

    # find all Q&A links
    links_file = "posts_links_vanphysics.txt"
    try:
        with open(links_file, "r") as link_file:
            print("Reading links from the file '{}'.".format(links_file))
            links = link_file.readlines()
            ques_links = [link.rstrip() for link in links]
            print("Total Q&A posts links: {}".format(len(ques_links)))
    except (FileNotFoundError ,IOError) :
        print("Getting the links for all the Q&A posts.")
        ques_links = []
        with open(links_file, "w") as link_file:
            for subcat_link in subcat_links:
                subcat_page = requests.get(subcat_link)
                subcat_soup = BeautifulSoup(subcat_page.text, 'html.parser')
                subcat_quns = subcat_soup.find_all("a", attrs = {"href": re.compile("listing.php*")})
                subcat_quns_links = [url_prefix + subcat_quns_link["href"] for subcat_quns_link in subcat_quns]
                for subcat_quns_link in subcat_quns_links:
                    ques_links.append(subcat_quns_link)
                    link_file.write("{}\n".format(subcat_quns_link))
                    if len(ques_links) % 50 == 0:
                        print("Total Q&A posts: {}".format(len(ques_links)), end='\r')
            print("Total Q&A posts links: {}".format(len(ques_links)))

    return ques_links
    
def get_qa_content(url_qa):
    page_qa = requests.get(url_qa)
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

links = get_qa_links(url)
content = get_qa_content(url_qa)

