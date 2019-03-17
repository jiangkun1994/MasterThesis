In [21]: result = {} 
    ...: for each_data in data: 
    ...:     truth = [each_data['title'].replace('_', ' ')] 
    ...:     for each_parags in each_data['paragraphs']: 
    ...:         for each_qa in each_parags['qas']: 
    ...:             question = each_qa['question'] 
    ...:             answer = [] 
    ...:             for each_answer in each_qa['answers']: 
    ...:                 answer.append(each_answer['text']) 
    ...:             result['question'] = question 
    ...:             result['answer'] = answer 
    ...:             result['truth'] = truth 
    ...:             with open('./metrics-squad.json', 'a') as f: 
    ...:                 f.write(json.dumps(result) + '\n')