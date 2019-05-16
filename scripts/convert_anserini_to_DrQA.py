import json 
with open('output.SQuAD.dev.txt') as f:
	all_doc_ids = [] 
	doc_ids = [] 
	counter = 0 
	for line in f: 
		each_doc_id = line.split('\t')[1] 
		doc_ids.append(each_doc_id) 
		counter += 1 
		if counter == 10: 
			all_doc_ids.append(doc_ids) 
			counter = 0 
			doc_ids = [] 

with open('SQuAD_for_eval.json', 'a') as fout: 
	fout.write(json.dumps(all_doc_ids))