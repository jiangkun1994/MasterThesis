################### for TF-IDF (basic) and Document Ranker
import json
n_paras = 10
precisions = []
mrrs = []
while True:
	test = []
	# for i in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-retrieval-T2000p-ptfidf.json'):
	for i in open('/home/jiangkun/MasterThesis/ParagraphRanker/ranker-trained-combined/re-rank/rerank-T' + str(n_paras) + 'para-squad.json'):
		test.append(json.loads(i))

	result = []
	for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-squad-paragraph.json'):
		result.append(json.loads(j))

	precision = []
	mrr = []
	for each_test in test:
		for each_result in result:
			if each_result['question'] == each_test['question']:
				retrieval = each_test['retrieval']
				truth = each_result['truth']
				break
		if retrieval == []:
			precision.append(0)
			mrr.append(0)
		else:
			first_retrieval = retrieval[0]
			if first_retrieval in truth:
			# if set(first_retrieval).intersection(set(truth)) != set():
				precision.append(1)
			else:
				precision.append(0)

			if set(retrieval).intersection(set(truth)) == set():
				mrr.append(0)
			else:
				for idx, text in enumerate(retrieval):
					if text in truth:
						mrr.append(1 / (idx + 1))
						break


	print('p@1: {}'.format(sum(precision) / len(precision)))
	print('MRR@5: {:.4f}'.format(sum(mrr) / len(mrr)))
	# print(precision)
	# print(mrr)
	print(len(precision))
	print(len(mrr))
	precisions.append(sum(precision) / len(precision))
	mrrs.append(sum(mrr) / len(mrr))
	n_paras += 1
	if n_paras > 20:
		break
# with open('/home/jiangkun/MasterThesis/ParagraphRanker/new-ranker/precision-mrr.txt', 'a') as f:
# 	f.write(json.dumps(precisions) + '\n')
# 	f.write(json.dumps(mrrs) +'\n')


################## for P-TF-IDF and P-Ranker
# import json

# test = []
# for i in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-retrieval-T2000p-ptfidf.json'):
# 	test.append(json.loads(i))

# result = []
# for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-squad-paragraph.json'):
# 	result.append(json.loads(j))

# precision = []
# mrr = []
# for each_test in test:
# 	num = 0
# 	for each_result in result:
# 		if each_result['question'] == each_test['question']:
# 			retrieval = each_test['retrieval']
# 			truth = each_result['truth']
# 			break
# 	if retrieval == []:
# 		precision.append(0)
# 		mrr.append(0)
# 	else:
# 		first_retrieval = retrieval[:72] ####### 51 for hotpotQA
# 		# first_retrieval.extend(retrieval[1][:30])
# 		if set(first_retrieval).intersection(set(truth)) == set():
# 			precision.append(0)
# 		else:
# 			precision.append(1)

# 		division = []
# 		division.append(retrieval[:72])
# 		division.append(retrieval[72:144])
# 		division.append(retrieval[144:216])
# 		division.append(retrieval[216:288])
# 		division.append(retrieval[288:360])

# 		for i, j in enumerate(division):
# 			if set(j).intersection(set(truth)) != set():
# 				mrr.append(1 / (i + 1))
# 				break
# 			else:
# 				num += 1
# 		if num == 5:
# 			mrr.append(0)
# 		# if first_retrieval in truth:
# 		# 	precision.append(1)
# 		# else:
# 		# 	precision.append(0)

# 		# if set(retrieval).intersection(set(truth)) == set():
# 		# 	mrr.append(0)
# 		# else:
# 		# 	for idx, text in enumerate(retrieval):
# 		# 		if text in truth:
# 		# 			mrr.append(1 / (idx + 1))
# 		# 			break



# print('p@1: {}'.format(sum(precision) / len(precision)))
# print('MRR@5: {}'.format(sum(mrr) / len(mrr)))
# # print("sum of precision: ", sum(precision))
# # print("sum of mrr: ", sum(mrr))
# print("length of precision: ", len(precision))
# print("length of mrr: ", len(mrr))
# print(precision)


######## precision for p-ranker
# import json

# test = []
# for i in open('/home/jiangkun/MasterThesis/ParagraphRanker/triviaQA-p-ranker-final-test.json'):
# 	test.append(json.loads(i))

# result = []
# for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-triviaQA.json'):
# 	result.append(json.loads(j))

# relevant = 0
# retrieved = 0

# for each_test in test:
# 	for i in each_test['truth']:
# 		retrieved += len(i)
# 	for each_result in result:
# 		if each_result['question'] == each_test['question']:
# 			truth = each_result['truth']
# 			break
# 	for i in each_test['truth']:
# 		for j in i:
# 			if j in truth:
# 				relevant += 1

# print('number of retrieved documents: ', retrieved)
# print('number of relevant documents: ', relevant)
# print('precision: ', relevant / retrieved)


######## precision for D-ranker
# import json

# test = []
# for i in open('/home/jiangkun/MasterThesis/ParagraphRanker/SQuAD-D-ranker-measure.json'):
# 	test.append(json.loads(i))

# result = []
# for j in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-squad.json'):
# 	result.append(json.loads(j))

# relevant = 0
# retrieved = 0

# for each_test in test:
# 	retrieved += len(each_test['truth'])
# 	for each_result in result:
# 		if each_result['question'] == each_test['question']:
# 			truth = each_result['truth']
# 			break
# 	for i in each_test['truth']:
# 		if i in truth:
# 			relevant += 1

# print('number of retrieved documents: ', retrieved)
# print('number of relevant documents: ', relevant)
# print('precision: ', relevant / retrieved)