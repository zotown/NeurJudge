import json
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from utils import Data_Process
import torch
import torch.nn as nn
from model import NeurJudge
import logging
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
# from word2vec import toembedding
random.seed(42)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# 读取JSON文件
def read_json(file_path):
    file_path = file_path
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    print("Micro precision\t%.4f" % micro_precision)
    print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    gen_result(res)

    return 0

#embedding = toembedding()
with open("./data/word2vec.json", 'r', encoding='utf-8') as f:
    embedding = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_dim = 256

model = NeurJudge(embedding)
model = model.to(device)
model_name = '_neural_judge_0'
print(model_name)
PATH = './data/'+model_name
# model.load_state_dict(torch.load(PATH))
model.to(device)

data = read_json('./data/sample10_result_gpt-4o-2024-08-06-editquery-1.json')

id2charge = json.load(open('./data/id2charge.json'))
time2id = json.load(open('./data/time2id.json'))

predictions = {"c": [], "q": []}
# 提取预测罪名和标准罪名标签
for record in data:
    predictions["q"].append(record["fact"])
    for candidate in record.get("candidate", []):
        predictions["c"].append(candidate["fact"])


#print(len(data_all))
dataloader = DataLoader(predictions["q"], batch_size = batch_size_dim, shuffle=False, num_workers=0, drop_last=False)

process = Data_Process()
legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()

legals,legals_len,arts,arts_sent_lent = legals.to(device),legals_len.to(device),arts.to(device),arts_sent_lent.to(device)
predictions_article = []
predictions_charge = []
predictions_time = []

true_article = []
true_charge = []
true_time = []


for step,batch in enumerate(tqdm(dataloader)):
    model.eval()
    charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
    documents = documents.to(device)
    
    sent_lent = sent_lent.to(device)
    true_article.extend(article_label.numpy())
    true_charge.extend(charge_label.numpy())
    true_time.extend(time_label.numpy())

    with torch.no_grad():
        charge_out,article_out,time_out = model(legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art,documents,sent_lent,process,device)
        
    charge_pred = charge_out.cpu().argmax(dim=1).numpy()
    article_pred = article_out.cpu().argmax(dim=1).numpy()
    time_pred = time_out.cpu().argmax(dim=1).numpy()

    predictions_article.extend(article_pred)
    predictions_charge.extend(charge_pred)
    predictions_time.extend(time_pred)

print('罪名')
eval_data_types(true_charge,predictions_charge,num_labels=115)
print('法条')
eval_data_types(true_article,predictions_article,num_labels=99)
print('刑期')
eval_data_types(true_time,predictions_time,num_labels=11)
