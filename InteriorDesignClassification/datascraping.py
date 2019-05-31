from bs4 import BeautifulSoup
import csv
import requests
import pandas as pd
from shutil import copyfile
from tqdm import tqdm
import pandas as pd


def extract_source(url):
  headers = {"User-Agent":"Mozilla/5.0"}
  source=requests.get(url, headers=headers).text
  return source

links = []
labels = []
df = pd.DataFrame(columns=["links","labels"])
count = 0

for i in tqdm(range(200000, 300000)):
  
  data = extract_source("https://havenly.com/boards/view/"+str(i))
  soup = BeautifulSoup(data, "lxml")
  srcs = [img['src'] for img in soup.find_all('img')]

  if "/img/icon/danger.png" in srcs:
    pass
  else:
    links.append(soup.find_all(name='meta')[-6])
    labels.append(soup.find_all(name='meta')[-3])
    
  if count % 1000 == 999:
    df = pd.DataFrame(zip(links,labels), columns = ["links","labels"])
    df.to_csv("data_"+str(count)+".csv", index=False)
    
  count = count + 1 
    
df = pd.DataFrame(zip(links,labels), columns = ["links","labels"])
df.to_csv("data_"+"final"+".csv", index=False)
print(len(links), len(labels))
