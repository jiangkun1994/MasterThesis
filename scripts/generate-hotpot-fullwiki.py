import json

test = []
for line in open('/home/jiangkun/MasterThesis/DrQA/hotpot-bridge-for-hotpot-reader-209p-ptfidf.json'):
	test.append(json.loads(line))

truth = []
for line in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/hotpot_fullwiki_test/hotpot_dev_fullwiki_v1_500_doc.json'):
	truth.append(json.loads(line))

final_result = []
for i, j in enumerate(truth):
	j['context'] = test[i]
	final_result.append(j)

with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/hotpot_fullwiki_test/hotpot-bridge-fullwiki-209p-ptfidf.json', 'w') as f:
	f.write(json.dumps(final_result))