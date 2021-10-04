#!/usr/bin/env python3
# Scrape the <https://van.physics.illinois.edu/qa/> website for Q&A posts.

import re
import requests
from bs4 import BeautifulSoup

url = "https://van.physics.illinois.edu/qa/"

page = requests.get(url)

# the way forward for getting all the posts is to try all the ids in the url:
# https://van.physics.illinois.edu/qa/listing.php?id=14422
