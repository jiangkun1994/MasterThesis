import json

n_paras = 21
while True:
	test = []
	for line in open('/home/jiangkun/MasterThesis/ParagraphRanker/ranker-trained-combined/content-for-hotpot-reader/content-for-hotpot-reader-rerank-' + str(n_paras) + 'para-ranker-combined.json'):
		test.append(json.loads(line))

	truth = []
	for line in open('/home/jiangkun/MasterThesis/DrQA/data/datasets/hotpot_fullwiki_test/hotpot_dev_fullwiki_v1_500_doc.json'):
		truth.append(json.loads(line))

	final_result = []
	for i, j in enumerate(truth):
		j['context'] = test[i]
		final_result.append(j)

	with open('/home/jiangkun/MasterThesis/ParagraphRanker/ranker-trained-combined/content-for-hotpot-reader/rerank-'+ str(n_paras) + 'para-list-ranker-combined.json', 'w') as f:
		f.write(json.dumps(final_result))

	n_paras += 1
	if n_paras > 100:
		break