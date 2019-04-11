###############  measure EM on one file
import json
import string
import regex as re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

test = []
for each_test in open('/home/jiangkun/MasterThesis/ParagraphRanker/prediction/check-semantic/SQuAD-v1.1-dev-500-doc-default-pipeline.preds'):
	test_data = json.loads(each_test)
	if test_data == []:
		test_data = 'aklsjdwilasd'
	else:
		test_data = test_data[0]['span']
	test.append(test_data)


truth = []
for each_truth in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-v1.1-dev-500-doc.txt'): ###### triviaQA-dev-500-doc  SQuAD-v1.1-dev-500-doc  hotpot_dev_distractor_v1_cp-500-doc
	truth_data = json.loads(each_truth)
	truth.append(truth_data['answer'])


exact_match = []
for i, j in enumerate(test):
	for each_idx, each_ground in enumerate(truth[i]):
		truth[i][each_idx] = normalize_answer(truth[i][each_idx])
	if normalize_answer(j) in truth[i]:
		exact_match.append(1)
	else:
		exact_match.append(0)

print('EM: ', sum(exact_match) / len(exact_match))
print('length of exact_match: ', len(exact_match))
print(sum(exact_match))
print(exact_match)








####################  measure EM on many files and plot the EM result  on SQuAD
# import json
# import string
# import regex as re
# import matplotlib.pyplot as plt
# import numpy as np

# def normalize_answer(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def exact_match_score(prediction, ground_truth):
#     """Check if the prediction is a (soft) exact match with the ground truth."""
#     return normalize_answer(prediction) == normalize_answer(ground_truth)


# truth = []
# for each_truth in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-v1.1-dev-500-doc.txt'): ###### triviaQA-dev-500-doc  SQuAD-v1.1-dev-500-doc  hotpot_dev_distractor_v1_cp-500-doc
# 	truth_data = json.loads(each_truth)
# 	truth.append(truth_data['answer'])

# final_EM = []
# for i in range(160):
# 	test = []
# 	num_paragraph = i+1
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i+1) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# for i in range(165, 301, 5):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)


# for i in range(300, 401, 50):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)


# for i in range(400, 951, 50):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# final_EM = np.array(final_EM)
# num_files_1 = np.arange(1, 161).tolist()
# num_files_2 = np.arange(165, 301, 5).tolist()
# num_files_3 = np.arange(300, 401, 50).tolist()
# num_files_4 = np.arange(400, 951, 50).tolist()
# num_files_1.extend(num_files_2)
# num_files_1.extend(num_files_3)
# num_files_1.extend(num_files_4)


# final_btfidf_EM = []
# for i in range(1, 14):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/SQuAD-v1.1-dev-500-doc-b-tfidf-' + str(i) + 'doc-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_btfidf_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# final_btfidf_EM = np.array(final_btfidf_EM)
# modified_btfidf_EM = np.array([0.2, 0.246, 0.26, 0.266, 0.26, 0.264, 0.262, 0.264, 0.258, 0.258, 0.254, 0.254, 0.256])
# num_files_btfidf = np.array([87, 167, 242, 315, 383, 450, 517, 588, 655, 720, 786, 853, 921])	

# xmax1 = num_files_1[np.argmax(final_EM)]
# ymax1 = final_EM.max()

# xmax2 = 4
# ymax2 = modified_btfidf_EM.max()

# text1 = "(paragraphs={}, EM={})".format(xmax1, ymax1)
# text2 = "(documents={}, EM={})".format(xmax2, ymax2)




# plt.plot(np.array(num_files_1), final_EM, 'r', label='Para-TFIDF')
# plt.plot(num_files_btfidf, modified_btfidf_EM, 'b^')
# plt.plot(num_files_btfidf, modified_btfidf_EM, 'g', label='Basic-TFIDF')
# plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '72paras)')
# plt.ylabel('Exact Match')
# plt.title('SQuAD')
# plt.legend(loc='best')
# plt.annotate(text1, xy=(18, 0.27), xytext=(100, 0.27),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.annotate(text2, xy=(315, 0.266), xytext=(320, 0.242),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.grid(True, linestyle='--')
# plt.xlim(left=1)
# plt.show()



###############  measure EM on many files and plot the EM result  on hotpotQA-Bridge and SQuAD
# import json
# import string
# import regex as re
# import numpy as np
# import matplotlib.pyplot as plt

# def normalize_answer(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def exact_match_score(prediction, ground_truth):
#     """Check if the prediction is a (soft) exact match with the ground truth."""
#     return normalize_answer(prediction) == normalize_answer(ground_truth)

# truth = []
# for each_truth in open('/home/jiangkun/MasterThesis/DrQA/SQuAD-v1.1-dev-500-doc.txt'): ###### triviaQA-dev-500-doc  SQuAD-v1.1-dev-500-doc  hotpot_dev_distractor_v1_cp-500-doc
# 	truth_data = json.loads(each_truth)
# 	truth.append(truth_data['answer'])


# final_EM = []
# for i in range(160):
# 	test = []
# 	num_paragraph = i+1
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i+1) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# for i in range(165, 301, 5):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)


# for i in range(300, 401, 50):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)


# for i in range(400, 951, 50):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/SQuAD-' + str(i) + '-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# final_EM = np.array(final_EM)
# num_files_1 = np.arange(1, 161).tolist()
# num_files_2 = np.arange(165, 301, 5).tolist()
# num_files_3 = np.arange(300, 401, 50).tolist()
# num_files_4 = np.arange(400, 951, 50).tolist()
# num_files_1.extend(num_files_2)
# num_files_1.extend(num_files_3)
# num_files_1.extend(num_files_4)


# final_btfidf_EM = []
# for i in range(1, 14):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/SQuAD-v1.1-dev-500-doc-b-tfidf-' + str(i) + 'doc-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth[i]):
# 			truth[i][each_idx] = normalize_answer(truth[i][each_idx])
# 		if normalize_answer(j) in truth[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_btfidf_EM.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# final_btfidf_EM = np.array(final_btfidf_EM)
# modified_btfidf_EM = np.array([0.2, 0.246, 0.26, 0.266, 0.26, 0.264, 0.262, 0.264, 0.258, 0.258, 0.254, 0.254, 0.256])
# num_files_btfidf = np.array([87, 167, 242, 315, 383, 450, 517, 588, 655, 720, 786, 853, 921])	

# xmax_squad_1 = num_files_1[np.argmax(final_EM)]
# ymax_squad_1 = final_EM.max()

# xmax_squad_2 = 4
# ymax_squad_2 = modified_btfidf_EM.max()

# text_squad_1 = "(paragraphs={}, EM={})".format(xmax_squad_1, ymax_squad_1)
# text_squad_2 = "(documents={}, EM={})".format(xmax_squad_2, ymax_squad_2)











# truth_bridge = []
# for each_truth in open('/home/jiangkun/MasterThesis/DrQA/hotpot_dev_distractor_v1_bridge-500-doc.txt'): ###### triviaQA-dev-500-doc  SQuAD-v1.1-dev-500-doc  hotpot_dev_distractor_v1_cp-500-doc
# 	truth_data = json.loads(each_truth)
# 	truth_bridge.append(truth_data['answer'])

# final_EM_bridge = []
# for i in range(1, 135):
# 	test = []
# 	num_paragraph = i
# 	for each_test in open('/home/jiangkun/MasterThesis/DrQA/prediction/paragraph-preds/hotpot-bridge-' + str(i) + 'p-all-rankbyme-pipeline.preds'):
# 		test_data = json.loads(each_test)
# 		if test_data == []:
# 			test_data = 'aklsjdwilasd'
# 		else:
# 			test_data = test_data[0]['span']
# 		test.append(test_data)

# 	exact_match = []
# 	for i, j in enumerate(test):
# 		for each_idx, each_ground in enumerate(truth_bridge[i]):
# 			truth_bridge[i][each_idx] = normalize_answer(truth_bridge[i][each_idx])
# 		if normalize_answer(j) in truth_bridge[i]:
# 			exact_match.append(1)
# 		else:
# 			exact_match.append(0)

# 	final_EM_bridge.append(sum(exact_match) / len(exact_match))
# 	print('EM: ', sum(exact_match) / len(exact_match))
# 	print('length of exact_match: ', len(exact_match))
# 	print(sum(exact_match))
# 	print('num of paragraph: ', num_paragraph)
# 	# print(exact_match)
# 	print('-'*70)

# num_files_para = np.arange(1, 135).tolist()
# num_files_para.extend([160, 209, 265, 331, 385, 440, 489, 537, 594, 645, 699, 754, 808])
# num_files_para = np.array(num_files_para)
# final_EM_para = np.array(final_EM_bridge).tolist()
# final_EM_para.extend([0.06, 0.06, 0.058, 0.056, 0.052, 0.052, 0.048, 0.048, 0.046, 0.044, 0.048, 0.044, 0.042])
# final_EM_para = np.array(final_EM_para)

# num_files_doc = np.array([51, 109, 160, 209, 265, 331, 385, 440, 489, 537, 594, 645, 699, 754, 808])
# final_EM_doc = np.array([0.056, 0.072, 0.076, 0.08, 0.08, 0.084, 0.084, 0.082, 0.082, 0.086, 0.084, 0.08, 0.078, 0.082, 0.082])


# # xmax1 = num_files_para[np.argmax(final_EM_para)]
# xmax1 = '10, 12 and 14-19'
# ymax1 = final_EM_para.max()

# xmax2 = 10
# ymax2 = final_EM_doc.max()

# text1 = "(paragraphs={}, EM={})".format(xmax1, ymax1)
# text2 = "(documents={}, EM={})".format(xmax2, ymax2)





# plt.subplot(1,2,1)
# plt.plot(np.array(num_files_1), final_EM, 'r', label='Para-TFIDF')
# plt.plot(num_files_btfidf, modified_btfidf_EM, 'b^')
# plt.plot(num_files_btfidf, modified_btfidf_EM, 'g', label='Basic-TFIDF')
# plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '72paras)')
# plt.ylabel('Exact Match')
# plt.title('SQuAD')
# plt.legend(loc='best')
# plt.annotate(text_squad_1, xy=(18, 0.27), xytext=(100, 0.27),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.annotate(text_squad_2, xy=(315, 0.266), xytext=(320, 0.255),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.grid(True, linestyle='--')
# plt.xlim(left=1)
# # plt.show()



# plt.subplot(1,2,2)
# plt.plot(num_files_para, final_EM_para, 'r', label='Para-TFIDF')
# plt.plot(num_files_doc, final_EM_doc, 'b^')
# plt.plot(num_files_doc, final_EM_doc, 'g', label='Basic-TFIDF')

# plt.annotate(text1, xy=(19, 0.102), xytext=(90, 0.10),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.annotate(text2, xy=(537, 0.086), xytext=(550, 0.092),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '54paras)')
# plt.ylabel('Exact Match')
# plt.title('HotpotQA-Bridge')
# plt.legend(loc='best')

# plt.grid(True, linestyle='--')
# plt.xlim(left=1)
# plt.show()