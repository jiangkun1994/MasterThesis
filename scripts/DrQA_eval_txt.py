b = {} 
a = [] 
for line in open('./test.txt'): 
    data = json.loads(line) 
    b['question'] = data['question'] 
    a.append(data['answer']) 
    a.append(data['answer']) 
    a.append(data['answer']) 
    b['answer'] = a 
    with open('./final.txt', 'a') as f: 
        f.write(json.dumps(b) + '\n') 
    a = [] 
    b = {} 