import json

test = []
for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/SQuAD-v1.1-dev-500-doc-default-pipeline.preds'):
	test_data = json.loads(each_test)
	test_data = test_data[0]['span']
	test.append(test_data)

truth = []
for each_truth in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-v1.1-dev-500-doc.txt'):
	truth_data = json.loads(each_truth)
	truth.append(truth_data['answer'])

exact_match = []
for i, j in enumerate(test):
	if j in truth[i]:
		exact_match.append(1)
	else:
		exact_match.append(0)

print('EM: ', sum(exact_match) / len(exact_match))
print('length of exact_match: ', len(exact_match))