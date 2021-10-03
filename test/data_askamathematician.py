#!/usr/bin/env python3
# Scrapte the <askamathematician.com> for Q&A and save it in a json.

import io
import re
import requests
from bs4 import BeautifulSoup

url = "https://www.askamathematician.com/
