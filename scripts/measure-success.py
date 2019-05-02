import json
import numpy as np
import matplotlib.pyplot as plt

n_paras = 10
final_success = []
while True:
	test = []
	# with open('/home/jiangkun/MasterThesis/DrQA/SQuAD-retrieval-T2000p-ptfidf.json') as f:
	with open('/home/jiangkun/MasterThesis/ParagraphRanker/ranker-trained-combined/re-rank/rerank-T' + str(n_paras) + 'para-hotpot.json') as f:
		for line in f:
			test_result = json.loads(line)
			test.append(test_result)

	truth = []
	with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-hotpot-paragraph.json') as f:
		for line in f:
			truth_result = json.loads(line)
			truth.append(truth_result)




	success = []
	for i, j in enumerate(test):
		for find_truth in truth:
			if j['question'] == find_truth['question']:
				if set(find_truth['truth']).intersection(set(j['retrieval'])) != set():
					success.append(1)
				else:
					success.append(0)
				break

	print(success)
	print(len(success))
	print("Success: ", sum(success) / len(success))
	final_success.append(sum(success) / len(success))

	n_paras += 1
	if n_paras > 100:
		break
with open('/home/jiangkun/MasterThesis/ParagraphRanker/ranker-trained-combined/success.txt', 'a') as f:
	f.write('Success: ' + json.dumps(final_success) + '\n')

# top_n_paras = np.array([51, 109, 160, 209, 265, 331, 385, 440, 489, 537, 594, 645, 699, 754, 808])
# final_success = np.array(final_success)
# # final_success_para = np.array([0.564, 0.632, 0.662, 0.674, 0.696, 0.714, 0.728, 0.742, 0.752, 0.754, 0.764, 0.766, 0.768])
# final_success_para = np.array([0.812, 0.846, 0.868, 0.872, 0.878, 0.884, 0.884, 0.888, 0.894, 0.894, 0.898, 0.90, 0.904, 0.906, 0.91])


# plt.plot(top_n_paras, final_success_para, 'r', label='Para-TFIDF')
# plt.plot(top_n_paras, final_success_para, 'o')
# plt.plot(top_n_paras, final_success, 'g', label='Basic-TFIDF')
# plt.plot(top_n_paras, final_success, '^')

# plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '54paras)')
# plt.ylabel('Success@n')
# plt.title('HotpotQA-Bridge')
# plt.grid(True, linestyle='--')
# plt.xlim(left=10)
# plt.legend(loc='best')
# plt.show()


###### measure success@n for Para-TFIDF
# import json


# para_test = []
# with open('/home/jiangkun/MasterThesis/DrQA/hotpot-bridge-retrieval-T2000p-ptfidf.json') as f:
# 	for line in f:
# 		para_test_result = json.loads(line)
# 		para_test.append(para_test_result)

# para_truth = []
# with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-hotpot-paragraph.json') as f:
# 	for line in f:
# 		para_truth_result = json.loads(line)
# 		para_truth.append(para_truth_result)

# success_para = []

# for i, j in enumerate(para_test):
# 	for find_truth in para_truth:
# 		if j['question'] == find_truth['question']:
# 			if set(find_truth['truth']).intersection(set(j['retrieval'][0:10])) != set():
# 				success_para.append(1)
# 			else:
# 				success_para.append(0)
# 			break

# print(success_para)
# print(len(success_para))
# print("success_para: ", sum(success_para) / len(success_para))
