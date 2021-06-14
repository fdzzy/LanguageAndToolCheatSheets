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

## Character ASCII
```python
>>> ord('a')
97
>>> chr(ord('a'))
'a'
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

## Date time
```python
from datetime import datetime
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta

today = date.today()
now = datetime.now()
date(2019, 9, 1)

now.strftime('%Y%m') # '201909'
now + timedelta(days=1, hours=8, minutes=15)
today + relativedelta(months=+1)
```

## Directory and file names
```python
>>> import os
>>> abs_path_name = '/root/path1/path2/filename.0.txt'
>>> os.path.dirname(abs_path_name)
'/root/path1/path2'
>>> os.path.basename(abs_path_name)
'filename.0.txt'
>>> os.path.splitext(abs_path_name)
('/root/path1/path2/filename.0', '.txt')
>>> os.path.dirname('test.txt')
''
>>> os.path.join(os.path.dirname('test.txt'), 'test2.txt')
'test2.txt'
```

## Exception handling
```python
try:
    # some code that might throw exceptions
    pass
except Exception as e:
    print(e)
finally:
    pass
```

## Object Equality
```python
class MyClass:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    def __eq__(self, other): 
        if not isinstance(other, MyClass):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.foo == other.foo and self.bar == other.bar

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.foo, self.bar))
```

## Encoding
```python
# -*- coding: utf-8 -*-
with open(infile, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()

with open(outfile, 'w', encoding='utf-8') as writer:
    writer.write(line)
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

## Argparse
```python
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input file.")
parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file.")
parser.add_argument("--n_process", default=0, type=int,
                        help="Number of processes to run.")
parser.add_argument("--debug", action='store_true',
                        help="debug mode")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
parser.add_argument("--outputs", required=True, nargs='+')
parser.add_argument("--outputs", required=True, nargs='*')
args = parser.parse_args()
args, unknown = parser.parse_known_args()
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

## JSON
```python
import json
# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'
# parse x:
y = json.loads(x)
# convert into JSON:
z = json.dumps(y)
# format the result
json.dumps(x, indent=4, separators=(". ", " = "))

def read_json_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as file:
        return json.load(file)

def write_to_json_file(filename, an_object, encoding='utf-8'):
    with open(filename, 'w', encoding=encoding) as file:
        json.dump(an_object, file, ensure_ascii=False, indent=4)

def json_beautify(infile, outfile, indent):
    with open(infile, 'r', encoding='utf-8') as reader, open(outfile, 'w', encoding='utf-8') as writer:
        obj = json.load(reader)
        json.dump(obj, writer, ensure_ascii=False, indent=indent)
```

## Argument Parsing
Using sys.argv
```python
import sys

print("the script has the name %s" % (sys.argv[0]))
# count the arguments
arguments = len(sys.argv) - 1
# output argument-wise
position = 1
while (arguments >= position):
    print("parameter %i: %s" % (position, sys.argv[position]))
    position = position + 1
```

Using argparse
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default=None, type=str, required=True,
                    help="The input file.")
parser.add_argument("--output_path", default=None, type=str, required=True,
                    help="The output file.")
parser.add_argument("--n_process", default=0, type=int,
                    help="Number of processes to run.")
parser.add_argument("--output_file_ext", default="asc", type=str,
                    help="If output is a directory, specify the file extension to use for the output files.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
args = parser.parse_args()
if args.do_train:
    train(args)
```

## LRU Cache
@functools.lru_cache(maxsize=128, typed=False)

## Multi-processing
```python
import os
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

def f(x):
    return x * x

def f2(x, num):
    return x ** num

p = Pool(5)
print(p.map(f, [1, 2, 3]))
print(p.map(partial(f2, num=2), [1, 2, 3]))

# Get number of processors
mp.cpu_count()
os.cpu_count()
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

## JSON
Text file format for storing collections of strings and numbers.
```python
import json
<str>    = json.dumps(<object>, ensure_ascii=True, indent=None)
<object> = json.loads(<str>)
```
Read Object from JSON file
```python
def read_json_file(filename):
    with open(filename, encoding='utf-8') as file:
        return json.load(file)
```
Write Object to JSON file
```python
def write_to_json_file(filename, an_object):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(an_object, file, ensure_ascii=False, indent=4)
```

## tqdm
```python
# regular
from tqdm import tqdm
# in jupyter notebook
from tqdm import tqdm_notebook as tqdm

for _ in tqdm(iterable, total=len(iterable), desc="description"):
    pass
```
Manual control of tqdm() updates:
```python
with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)

with tqdm() as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update()

pbar = tqdm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()
```

## Queue
```python
>>> from queue import Queue
>>> a = Queue()
>>> a.put(1)
>>> a.put(2)
>>> a.put(3)
>>> a.get()
1
>>> a.empty()
False
>>> a.qsize()
2
>>> a.get_nowait()
2

import threading, queue

q = queue.Queue()

def worker():
    while True:
        item = q.get()
        print(f'Working on {item}')
        print(f'Finished {item}')
        q.task_done()

# turn-on the worker thread
threading.Thread(target=worker, daemon=True).start()

# send thirty task requests to the worker
for item in range(30):
    q.put(item)
print('All task requests sent\n', end='')

# block until all tasks are done
q.join()
print('All work completed')

>>> from collections import deque
>>> queue = deque([1,2])
>>> queue.popleft()
1
>>> queue
deque([2])
>>> queue.append(3)
>>> queue
deque([2, 3])
>>> len(queue)
2

```

## Priority Queue
```python
>>> from queue import PriorityQueue
>>> a = PriorityQueue()
>>> a.put(2)
>>> a.put(3)
>>> a.put(1)
>>> a.put(10)
>>> a.get()
1
>>> a.empty()
False
>>> a.qsize()
3
>>> a.get()
2
```

# Data science
## Numpy

## Matplotlib
```python
import matplotlib.pyplot as plt
%matplotlib inline
```

## Pandas
Read TSV
```python
import pandas as pd
import csv
df = pd.read_csv(data_path, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, names=["column1", "column2"])
# peek
df.describe()
df.head()
# add a new column based on old column
df['NewColumn'] = df['column1'].map(str).map(lambda x: x + "new")

# iterate rows
for row in tqdm(df.itertuples(), total=len(df)):
    print(row.NewColumn)

# select columns
df_new = df[['column1', 'column2']]

# output to tsv
df.to_csv(output_path, index=False, header=False, encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)

# select rows using predicates
df_wav[df_wav['duration'] < 20]

# percentile
df['Length'].quantile(0.95)

# show histogram
df_wav['duration'].hist()

# value counts
df['column1'].value_counts()
```

## Scikit-learn
Confusion Matrix
https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
```python
from sklearn.metrics import confusion_matrix

y_true = ['Cat', 'Dog', 'Rabbit', 'Cat', 'Cat', 'Rabbit']
y_pred = ['Dog', 'Dog', 'Rabbit', 'Dog', 'Dog', 'Rabbit']

classes=['Cat', 'Dog', 'Rabbit']

confusion_matrix(y_true, y_pred, labels=['Cat', 'Dog', 'Rabbit'])

array([[0, 3, 0],
       [0, 1, 0],
       [0, 0, 2]])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=['Cat', 'Dog', 'Rabbit'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Cat', 'Dog', 'Rabbit'],
                      title='Confusion matrix, without normalization')
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

Auto reload
```python
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
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

NLTK Stanford Parser
```python
from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = '/mnt/e/software/stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_models_jar = '/mnt/e/software/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar, java_options=' -mx4G')

sentences = ['They have nice dessert', 'Its camera is great.', 'I love their fries.', 'iPhone is the best cellphone.', 'I like Indian food.', 'Their spring roll is great.']

for sent in sentences:
    print([list(parse.triples()) for parse in dependency_parser.raw_parse(sent)])
```

## WordNet
http://www.nltk.org/howto/wordnet.html
```python
from nltk.corpus import wordnet as wn
wn.synsets('screen')
wn.synsets('operating_system')
```
Synsets
```python
>>> dog = wn.synset('dog.n.01')
>>> dog.hypernyms()
[Synset('canine.n.02'), Synset('domestic_animal.n.01')]
>>> dog.hyponyms()  # doctest: +ELLIPSIS
[Synset('basenji.n.01'), Synset('corgi.n.01'), Synset('cur.n.01'), Synset('dalmatian.n.02'), ...]
>>> dog.member_holonyms()
[Synset('canis.n.01'), Synset('pack.n.06')]
>>> dog.root_hypernyms()
[Synset('entity.n.01')]
>>> wn.synset('dog.n.01').lowest_common_hypernyms(wn.synset('cat.n.01'))
[Synset('carnivore.n.01')]
```
Similarity
```python
>>> dog = wn.synset('dog.n.01')
>>> cat = wn.synset('cat.n.01')
>>> dog.path_similarity(cat)  # doctest: +ELLIPSIS
0.2...
>>> hit.path_similarity(slap)  # doctest: +ELLIPSIS
0.142...
>>> wn.path_similarity(hit, slap)  # doctest: +ELLIPSIS
0.142...
>>> print(hit.path_similarity(slap, simulate_root=False))
None
>>> print(wn.path_similarity(hit, slap, simulate_root=False))
None
```

## Spacy
https://spacy.io/usage/spacy-101
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

### Dependency parsing
https://spacy.io/usage/linguistic-features#dependency-parse

https://explosion.ai/demos/displacy
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
```

# Multimedia
## Audio
Trim silence
```python
from pydub import AudioSegment

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms

def trim_silence(inputfn, outputfn, format, verbose=True):
    sound = AudioSegment.from_file(inputfn, format=format)
    start_trim = detect_leading_silence(sound, silence_threshold=-40)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    original_len = duration / 1000.0
    new_len = original_len
    if (start_trim >= duration) or (end_trim <= 0):
        new_len = 0
    else:
        trimmed_sound = sound[start_trim:duration-end_trim]
        original_len = duration / 1000.0
        new_len = len(trimmed_sound) / 1000.0
        trimmed_sound.export(outputfn, format=format)
    if verbose:
        print("Before trim: %f (seconds)\nAfter trim: %f (seconds)" % (original_len, new_len))
    return new_len
```

Get audio info
```python
import soundfile as sf
def try_voice_info(filename):
    try:
        return sf.info(filename)
    except:
        return None

df_wav['Info'] = df_wav['Filename'].map(try_voice_info)
df_wav = df_wav[df_wav['Info'].notnull()].copy()

for col in ['channels', 'duration', 'samplerate', 'endian', 'format', 'format_info', 'subtype', 'subtype_info', 'frames']:
    df_wav[col] = df_wav['Info'].map(lambda x: getattr(x, col))
```

Resample
```python
import soundfile as sf
from librosa.core import resample

target_rate = 16000
data, rate  = sf.read(filename)
data = resample(data, rate, target_rate)
sf.write(trainFilename, data, samplerate=target_rate, subtype='PCM_16', format='WAV')
```

## Image

# Web
## Requests
Post audio file
```python
import requests

data = open('voice.wav', 'rb').read()
res = requests.post(url='https://example.com/api/audio/classify', data=data, headers={'Content-Type': 'application/octet-stream'})

url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
html = requests.get(url).text
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

# attribute
anchors = doc.find_all('a')
anchors[0].attrs['href']
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

## Custom code

```python
class Writer:
    def __init__(self, file, encoding='utf-8'):
        self.file = file
        self.encoding = encoding
        self.writer = None
    
    def __enter__(self):
        self.writer = open(self.file, 'w', encoding=self.encoding)
        return self.writer

    def __exit__(self, type, value, traceback):
        if self.writer is not None:
            self.writer.close()

class TimedBlock:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        print(f'Start running block {self.name}...')
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        end = time.time()
        print(f'Finished running block {self.name} in {end - self.start} seconds!')
```