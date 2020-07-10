import json

a = []
for i in range(10):
    a.append({})
    a[i]['item-1'] = i
    a[i]['item-2'] = i

with open("data_file.json","w") as write_file:
    json.dump(a,write_file)
