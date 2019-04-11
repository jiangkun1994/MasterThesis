import matplotlib.pyplot as plt
import numpy as np

num_para_squad = np.array([87, 167, 242, 315, 383, 450, 517, 588, 655, 720])
ans_recall_doc_squad = np.array([0.574, 0.7, 0.76, 0.796, 0.808, 0.82, 0.838, 0.842, 0.846, 0.85])
ans_recall_para_squad = np.array([0.79, 0.858, 0.888, 0.898, 0.91, 0.916, 0.922, 0.932, 0.932, 0.936])



num_para_bridge = np.array([51, 109, 160, 209, 265, 331, 385, 440, 489, 537, 594, 645, 699, 754, 808])
ans_recall_doc_bridge = np.array([0.316, 0.424, 0.50, 0.54, 0.562, 0.578, 0.594, 0.604, 0.61, 0.62, 0.632, 0.644, 0.65, 0.654, 0.662])
ans_recall_para_bridge = np.array([0.576, 0.63, 0.666, 0.682, 0.696, 0.712, 0.722, 0.732, 0.744, 0.746, 0.752, 0.752, 0.758, 0.764, 0.766])



plt.subplot(1,2,1)
plt.plot(num_para_squad, ans_recall_para_squad, 'r', label='Para-TFIDF')
plt.plot(num_para_squad, ans_recall_para_squad, 'o')
plt.plot(num_para_squad, ans_recall_doc_squad, 'g', label='Basic-TFIDF')
plt.plot(num_para_squad, ans_recall_doc_squad, '^')

plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '72paras)')
plt.ylabel('Answer Recall')
plt.title('SQuAD')
plt.legend(loc='best')
plt.grid(True, linestyle='--')


plt.subplot(1,2,2)
plt.plot(num_para_bridge, ans_recall_para_bridge, 'r', label='Para-TFIDF')
plt.plot(num_para_bridge, ans_recall_para_bridge, 'o')
plt.plot(num_para_bridge, ans_recall_doc_bridge, 'g', label='Basic-TFIDF')
plt.plot(num_para_bridge, ans_recall_doc_bridge, '^')


plt.xlabel('Number of Paragraphs (1doc' + r'$\approx$' + '54paras)')
plt.ylabel('Answer Recall')
plt.title('HotpotQA-Bridge')
plt.legend(loc='best')

plt.grid(True, linestyle='--')
# plt.xlim(left=1)
plt.show()