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
    for subcat_quns_link in subcat_quns_links:
        print(subcat_quns_link)
        ques_links.append(subcat_quns_link)
        print("Total Q&A posts: {}".format(len(ques_links)))
    

