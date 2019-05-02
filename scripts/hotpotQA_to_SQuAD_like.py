import json
from tqdm import tqdm

final_result = {}
final_each_result = []
with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/hotpot_dev_distractor_v1.json') as f:
	data = json.load(f)
	for each_data in tqdm(data):
		if each_data['answer'] == 'yes' or each_data['answer'] == 'no':
			continue
		else:
			for each_sp in each_data['supporting_facts']:
				final_each_data = {}
				final_each_data['title'] = each_sp[0] # give each supporting fact document
				for each_context in each_data['context']:
					if each_context[0] == each_sp[0]: # find the sp context
						true_context = ' '.join(each_context[1]) # combine the sentence
						final_each_data['paragraphs'] = [{'context': true_context, 'qas': []}]
						break
					else:
						continue
				each_qas = {}
				if each_data['answer'] in true_context:
					for word_idx, word in enumerate(true_context):
						if true_context[word_idx: word_idx+len(each_data['answer'])] == each_data['answer']:
							answer_start = word_idx
				else:
					answer_start = 0
				each_qas['answers'] = [{'answer_start': answer_start, 'text': each_data['answer']}]
				each_qas['question'] = each_data['question']
				each_qas['id'] = each_data['_id']
				final_each_data['paragraphs'][0]['qas'].append(each_qas)
				final_each_result.append(final_each_data)

final_result['data'] = final_each_result
final_result['version'] = '1.2'

with open('/home/jiangkun/MasterThesis/ParagraphRanker/hotpot_dev_squad_like.json', 'w') as f:
	f.write(json.dumps(final_result))
