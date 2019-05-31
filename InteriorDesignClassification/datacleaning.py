import pandas as pd
import wget
data = pd.read_csv("datafile.csv",header=None)
data.columns = ["links", "labels"]
url = data['links'][0]
temp = data['labels'][0]
label = temp.split(',')[0]

labels = data['labels'].values
links = data['links'].values
lb = []
lnk = []
indices = []
list_classes = ['Classic','Modern','Glam','Industrial','Traditional', 'Coastal', 'Global', 'Preppy','Rustic','Transitional', 'Farmhouse','Bohemian', 'Midcentury Modern','Scandinavian','Eclectic','Minimal']
for i in range(len(labels)):
    
    if labels[i].split(',')[0] in list_classes:
        lb.append(labels[i].split(',')[0])
    else:    
        indices.append(i)

for i in range(len(lb)):
    lnk.append("file"+str(i)+".jpg")

print(data.shape)
data = data.drop(data.index[indices])
print(data.shape)
print(len(lb),len(lnk))
data = data.drop("labels", axis= 1)
data['labels'] = lb
data.to_csv("dataset_dropped", index=False)

df = pd.DataFrame(columns=['filename','labels'])
df["filename"] = lnk
df['labels'] = lb
print(df.head())
print(df['labels'].unique(), len(df['labels'].unique()))
print(df['labels'].value_counts())
df.to_csv("dataset.csv", index=False)


