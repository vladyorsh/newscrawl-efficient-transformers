import os
import requests
import argparse
from bs4 import BeautifulSoup

#Download document-split versions of the dataset
parser = argparse.ArgumentParser()
parser.add_argument('-l','--langs', nargs='+', help='Langs to download', required=False, default=['cs', 'en'])
parser.add_argument('-y','--years', nargs='+', help='Years to download', required=False, default=None)
parser.add_argument('-t','--target', help='Where to store files', required=False, default='.')
parser.add_argument('-u','--unpack', type=bool, help='Unpack .gz?', required=False, default=True)
args = parser.parse_args()

user = 'newscrawl'
password = 'acrawl4me'
root = f'https://{user}:{password}@data.statmt.org/news-crawl/doc/'

langs = args.langs
years = args.years
target= args.target
unpack= args.unpack

for lang in langs:
  lang = os.path.join(root, lang)
  r = requests.get(lang)
  html = BeautifulSoup(r.text, features='lxml')
  for a in html.find_all('a'):
    href = a["href"]
    if href.endswith('.gz'):
      link = os.path.join(lang, href)
      if years is not None:
        year = link.split('/')[-1]
        year = year.split('.')[1]
        if year not in years:
          continue
      link = 'https://' + link.split('@')[1]
      os.system(f"wget -cN -P \"{target}\" --user {user} --password {password} \"{link}\"")

if unpack:
  for f in os.listdir(target):
      if f.endswith('.gz'):
        f = os.path.join(target, f)
        os.system(f'gzip -df {f}')