# References
https://github.com/gto76/python-cheatsheet

# Fundamentals

## Main
```python
if __name__ == '__main__': # Runs main() if file wasn't imported.
    main()
```

## List

```python
list1 = []
list1.append('a') # ['a']

list2 = ['b', 'c']
list1.extend(list2) # ['a', 'b', 'c']
```

## Dictionary

```python
<view> = <dict>.keys()                          # Coll. of keys that reflects changes.
<view> = <dict>.values()                        # Coll. of values that reflects changes.
<view> = <dict>.items()                         # Coll. of key-value tuples.

value  = <dict>.get(key, default=None)          # Returns default if key does not exist.
value  = <dict>.setdefault(key, default=None)   # Same, but also adds default to dict.
<dict> = collections.defaultdict(<type>)        # Creates a dict with default value of type.
<dict> = collections.defaultdict(lambda: 1)     # Creates a dict with default value 1.

<dict>.update(<dict>)
<dict> = dict(<collection>)                     # Creates a dict from coll. of key-value pairs.
<dict> = dict(zip(keys, values))                # Creates a dict from two collections.
<dict> = dict.fromkeys(keys [, value])          # Creates a dict from collection of keys.

value = <dict>.pop(key)                         # Removes item or raises KeyError.
{k: v for k, v in <dict>.items() if k in keys}  # Filters dictionary by keys.
```

Sort dictionary by value
```python
for k,v in sorted(<dict>.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
    print(k, v)
```

## Counter
```python
>>> from collections import Counter
>>> colors = ['red', 'blue', 'yellow', 'blue', 'red', 'blue']
>>> counter = Counter(colors)
Counter({'blue': 3, 'red': 2, 'yellow': 1})
>>> counter.most_common()[0]
('blue', 3)
```

## Regex
```python
import re
<str>   = re.sub(<regex>, new, text, count=0)  # Substitutes all occurrences.
<list>  = re.findall(<regex>, text)            # Returns all occurrences.
<list>  = re.split(<regex>, text, maxsplit=0)  # Use brackets in regex to keep the matches.
<Match> = re.search(<regex>, text)             # Searches for first occurrence of pattern.
<Match> = re.match(<regex>, text)              # Searches only at the beginning of the text.
<iter>  = re.finditer(<regex>, text)           # Returns all occurrences as match objects.
```

- Search() and match() return None if there are no matches.
- Argument 'flags=re.IGNORECASE' can be used with all functions.
- Argument 'flags=re.MULTILINE' makes '^' and '$' match the start/end of each line.
- Argument 'flags=re.DOTALL' makes dot also accept newline.
- Use r'\1' or '\\1' for backreference.
- Add '?' after an operator to make it non-greedy.

Match Object
```python
<str>   = <Match>.group()   # Whole match. Also group(0).
<str>   = <Match>.group(1)  # Part in first bracket.
<tuple> = <Match>.groups()  # All bracketed parts.
<int>   = <Match>.start()   # Start index of a match.
<int>   = <Match>.end()     # Exclusive end index of a match.
```

## Multi-processing
```python
from multiprocessing import Pool
from functools import partial

def f(x):
    return x * x

def f2(x, num):
    return x ** num

p = Pool(5)
print(p.map(f, [1, 2, 3]))
print(p.map(partial(f2, num=2), [1, 2, 3]))
```

## XML
Refer to https://docs.python.org/3/library/xml.etree.elementtree.html
```python
import xml.etree.ElementTree as ET
tree = ET.parse('country_data.xml')
root = tree.getroot()

for neighbor in root.iter('neighbor'):
    print(neighbor.attrib)
```

# Data science
## Numpy

## Pandas
Read TSV
```python
import pandas as pd
import csv
df = pd.read_csv(data_path, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, names=["column1", "column2"])
```

## Jupyter Setup
Refer to https://jupyter-notebook.readthedocs.io/en/stable/public_server.html
- Setup `virtualenv`
- Install jupyter
```bash
$ pip install jupyter
```
- Generate jupyter config:
```bash
$ jupyter notebook --generate-config
```
The default location for this file is your Jupyter folder located in your home directory:

Windows: C:\Users\USERNAME\.jupyter\jupyter_notebook_config.py

OS X: /Users/USERNAME/.jupyter/jupyter_notebook_config.py

Linux: /home/USERNAME/.jupyter/jupyter_notebook_config.py

- Set password
```bash
$ jupyter notebook password # jupyter notebook password will prompt you for your password and record the hashed password in your jupyter_notebook_config.json
```

- Start jupyter
```bash
$ jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --NotebookApp.allow_password_change=False
```

# NLP
## NLTK
Tokenization
```python
tokens = nltk.word_tokenize(text)
```
Chunking
```python
>>> sent = "Great laptop that offers many great features!"
>>> nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))
Tree('S', [Tree('GPE', [('Great', 'NNP')]), ('laptop', 'VBZ'), ('that', 'IN'), ('offers', 'VBZ'), ('many', 'JJ'), ('great', 'JJ'), ('features', 'NNS'), ('!', '.')])
```

## Spacy
```python
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
```

# Multimedia
## Audio

## Image

# Web
## Requests
Post audio file
```python
import requests

data = open('voice.wav', 'rb').read()
res = requests.post(url='https://example.com/api/audio/classify', data=data, headers={'Content-Type': 'application/octet-stream'})
```

## Scraping
Scrapes Python's logo, URL and version number from Wikipedia page:
```python
# $ pip3 install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup
url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
html = requests.get(url).text
doc = BeautifulSoup(html, 'html.parser')
table = doc.find('table', class_='infobox vevent')
rows = table.find_all('tr')
link = rows[11].find('a')['href']
ver = rows[6].find('div').text.split()[0]
url_i = rows[0].find('img')['src']
image = requests.get(f'https:{url_i}').content
with open('test.png', 'wb') as file:
    file.write(image)
print(link, ver)
```

## Selenium
Example
```python
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

class Scraper:
    username = ''
    password = ''
    browser = None

    def __init__(self, username=UserName, password=Password):
        self.username = username
        self.password = password
        
    def Login(self):
        binary = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe')
        self.browser = webdriver.Firefox(firefox_binary=binary)
        self.browser.get('https://www.example.com/')
        
        UN = self.browser.find_element_by_id('email')
        UN.send_keys(self.username)
        
        PS = self.browser.find_element_by_id('pass')
        PS.send_keys(self.password)
        
        LI = self.browser.find_element_by_id('loginbutton')
        LI.click()
        
    def Quit(self):
        if self.browser is not None:
            self.browser.quit()
        
    def GetHtml(self, url):
        self.browser.get(url)
        html_source = self.browser.page_source
        return html_source
        
    def OpenUrl(self, url):
        self.browser.get(url)
        
    def ScrollDown(self):
        self.browser.set_script_timeout(300)
        self.browser.execute_script("window.scrollTo(0,Math.max(document.documentElement.scrollHeight," + "document.body.scrollHeight,document.documentElement.clientHeight));")
    
    def GetCurrentPageSource(self):
        return self.browser.page_source
        
    def GetCurrentUrl(self):
        return self.browser.current_url
```

## Flask