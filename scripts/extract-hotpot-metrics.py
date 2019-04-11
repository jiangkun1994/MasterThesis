# In [30]: result = {} 
#     ...: for each_data in data: 
#     ...:     question = each_data['question'] 
#     ...:     answer = [each_data['answer']] 
#     ...:     truth = [] 
#     ...:     for i in each_data['supporting_facts']: 
#     ...:         truth.append(i[0]) 
#     ...:     result['question'] = question 
#     ...:     result['answer'] = answer 
#     ...:     result['truth'] = truth 
#     ...:     with open('./metrics-hotpot.json', 'a') as f: 
#     ...:         f.write(json.dumps(result) + '\n')





import json

with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/hotpot_dev_distractor_v1.json') as f:
    data = json.load(f)

result = {}
for each_data in data:
    question = each_data['question']
    answer = [each_data['answer']]
    truth = []
    for i in each_data['supporting_facts']:
        truth.append(i[0] + '-' + str(1))
    result['question'] = question
    result['answer'] = answer
    result['truth'] = truth
    with open('/home/jiangkun/MasterThesis/DrQA/data/datasets/metrics-hotpot-paragraph.json', 'a') as f:
        f.write(json.dumps(result) + '\n')