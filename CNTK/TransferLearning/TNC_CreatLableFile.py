import os
import pandas as pd

base_folder = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_folder, "DataSets")
source_path = os.path.join(datasets_dir, "ByLabelsCleanUp")

df = pd.DataFrame(columns=['FileName', 'Format', 'Folder', 'Category', 'Label'])
index = 0
for root, dirs, files in os.walk(source_path):
    for name in files:
        print(os.path.join(root,name))
        print("Name        = ", os.path.basename(name))
        print("File Name   = ", name[:-4])
        print("Folder Name = ", name[:name.rfind('-')])
        head, tail = os.path.split(root)
        print("Label       = ", tail)
        print("==================================================")
        print("")
        df.loc[index] = [name[:-4], 'JPG', name[:name.rfind('-')], tail, tail]
        index += 1

print(df.shape)
df.to_csv("RawDataLabel.csv", index=False, encoding='utf-8-sig')
g = pd.DataFrame({'Count' : df.groupby([ "Label"] ).size()}).reset_index()
g = g.sort_values(by=["Count"], ascending=False)
print(g)
