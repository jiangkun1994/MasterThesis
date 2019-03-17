################### for TF-IDF (basic) and Document Ranker
# import json

# test = []
# for i in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/SQuAD-test-p-tfidf-measure-1.json'):
# 	test.append(json.loads(i))

# result = []
# for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-squad.json'):
# 	result.append(json.loads(j))

# precision = []
# mrr = []
# for each_test in test:
# 	num = 0
# 	for each_result in result:
# 		if each_result['question'] == each_test['question']:
# 			retrieval = each_test['truth']
# 			truth = each_result['truth']
# 			break
# 	if retrieval == []:
# 		precision.append(0)
# 		mrr.append(0)
# 	else:
# 		first_retrieval = retrieval[0]
# 		if first_retrieval in truth:
# 			precision.append(1)
# 		else:
# 			precision.append(0)

# 		if set(retrieval).intersection(set(truth)) == set():
# 			mrr.append(0)
# 		else:
# 			for idx, text in enumerate(retrieval):
# 				if text in truth:
# 					mrr.append(1 / (idx + 1))
# 					break



# print('p@1: {}'.format(sum(precision) / len(precision)))
# print('MRR@5: {}'.format(sum(mrr) / len(mrr)))
# # print(precision)
# # print(mrr)
# print(len(precision))
# print(len(mrr))


################## for P-TF-IDF and P-Ranker
import json

test = []
for i in open('/home/jiangkun/MasterThesis/ParagraphRanker/hotpot-bridge-test-P-ranker-measure.json'):
	test.append(json.loads(i))

result = []
for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-hotpot.json'):
	result.append(json.loads(j))

precision = []
mrr = []
for each_test in test:
	num = 0
	for each_result in result:
		if each_result['question'] == each_test['question']:
			retrieval = each_test['truth']
			truth = each_result['truth']
			break
	if retrieval == []:
		precision.append(0)
		mrr.append(0)
	else:
		first_retrieval = retrieval[0]
		if set(first_retrieval).intersection(set(truth)) == set():
			precision.append(0)
		else:
			precision.append(1)

		for i, j in enumerate(retrieval):
			if set(j).intersection(set(truth)) != set():
				mrr.append(1 / (i + 1))
				break
			else:
				num += 1
		if num == 5:
			mrr.append(0)
		# if first_retrieval in truth:
		# 	precision.append(1)
		# else:
		# 	precision.append(0)

		# if set(retrieval).intersection(set(truth)) == set():
		# 	mrr.append(0)
		# else:
		# 	for idx, text in enumerate(retrieval):
		# 		if text in truth:
		# 			mrr.append(1 / (idx + 1))
		# 			break



print('p@1: {}'.format(sum(precision) / len(precision)))
print('MRR@5: {}'.format(sum(mrr) / len(mrr)))
# print("sum of precision: ", sum(precision))
# print("sum of mrr: ", sum(mrr))
print("length of precision: ", len(precision))
print("length of mrr: ", len(mrr))