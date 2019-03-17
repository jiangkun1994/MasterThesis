In [17]: result = {} 
    ...: for each_data in datapoint: 
    ...:     question = each_data['Question'] 
    ...:     answer = each_data['Answer']['Aliases'] 
    ...:     truth = [] 
    ...:     for i in each_data['EntityPages']: 
    ...:         truth.append(i['Title']) 
    ...:     result['question'] = question 
    ...:     result['answer'] = answer 
    ...:     result['truth'] = truth 
    ...:     with open('./metrics-triviaQA.json', 'a') as f: 
    ...:         f.write(json.dumps(result) + '\n')