"""
Retrieving 1000 QA pairs from each of the four datasets I used for error analysis
"""

import json



txt = ['../../DrQA/data/datasets/SQuAD-v1.1-dev.txt', '../../DrQA/data/datasets/triviaQA-dev.txt', 
       '../../DrQA/data/datasets/hotpot_dev_distractor_v1_bridge.txt', '../../DrQA/data/datasets/hotpot_dev_distractor_v1_cp.txt']
for t in txt:
	num = 0
	for line in open(t):
		num += 1
		if num > 1000:
			break
		dataset = json.loads(line)
		with open("../../DrQA/data/datasets/error_analysis_4000_label.txt", 'a') as f:
			if t == '../../DrQA/data/datasets/SQuAD-v1.1-dev.txt':
			    f.write(json.dumps(dataset) + '\t' + 'squad' + '\n')
			elif t == '../../DrQA/data/datasets/triviaQA-dev.txt':
				f.write(json.dumps(dataset) + '\t' + 'trivia' + '\n')
			elif t == '../../DrQA/data/datasets/hotpot_dev_distractor_v1_bridge.txt':
				f.write(json.dumps(dataset) + '\t' + 'bridge' + '\n')
			else:
				f.write(json.dumps(dataset) + '\t' + 'cp' + '\n')
