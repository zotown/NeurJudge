# @Time : 2024/9/1 22:42 
# @Author : LiuHuanghai
# @File : exchangeid.json.py 
# @Project: PyCharm
import json

# Load the original article2id.json
with open('charge2id.json', 'r') as file:
    article2id = json.load(file)

# Reverse the keys and values
id2article = {v: k for k, v in article2id.items()}

# Save the reversed dictionary to id2article.json
with open('id2charge.json', 'w') as file:
    json.dump(id2article, file, indent=4)