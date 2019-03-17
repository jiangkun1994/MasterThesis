import numpy as np
import json


random_int = np.random.randint(7993, size=1200)
random_int = set(random_int)
print(len(random_int))

content = []
for line in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/triviaQA-dev.txt'):
	result = json.loads(line)
	content.append(result)

dict_content = {}
for i, j in enumerate(content):
	dict_content[i] = j

for number in random_int:
	with open('/home/jiangkun/MasterThesis/DrQA/triviaQA-dev-1000-1.txt', 'a') as f:
		f.write(json.dumps(dict_content[number]) + '\n')