import argparse
import logging
from decimal import Decimal
import time
from time import perf_counter, process_time

import numpy as np
from sklearn.metrics import f1_score  
import torch
from torch.utils import data
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, AdamW

from deepspeed.profiling.flops_profiler import FlopsProfiler, get_model_profile
from codecarbon import OfflineEmissionsTracker

# Measure carbon emissions
tracker = OfflineEmissionsTracker(country_iso_code="CAN")
tracker.start()


# Measure time:
start = time.time()
relative_start = perf_counter()
#process_start = process_time()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default='data/train.txt', help='Path to training data')
parser.add_argument('--test', default='data/test_simple.txt', help='Path to test data')
args = parser.parse_args()

train_path = args.train
test_path = args.test

def read_files(file_path):
    sents, labels = [],[]
    with open(file_path,'r') as lines:
        for line in lines:
            label, sent = line.strip().split('\t')
            labels.append(int(label))
            sents.append(sent)

    return sents, labels

train_texts, train_tags = read_files(train_path)
test_texts, test_tags = read_files(test_path)

label_dict = {"whale":0, "boat":1, "man":2}   

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

# Subword-tokenize training and test data:
train_encodings = tokenizer(train_texts, is_split_into_words=False, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=False, padding=True, truncation=True)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create datasets:
train_dataset = SimpleDataset(train_encodings, train_tags)
test_dataset = SimpleDataset(test_encodings, test_tags)

# Enable using a GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using {device}")

token_model = AutoModelForTokenClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=len(set(label_dict.keys())))
#print(token_model)

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=len(set(label_dict.keys())))

#print(model)

prof = FlopsProfiler(model)

model.to(device)
model.train() # Enables the update of weights

prof.start_profile()

# PyTorch DataLoader handles batching and other routines for Datasets during training:
train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Our optimizer
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(20):
    total_loss = 0.0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels) #Forward pass
        loss = outputs[0]
        loss.backward() #Backward pass
        optim.step()
        
        total_loss += loss
    
    logging.info(f"Epoch {epoch}: training loss: {total_loss}")

logging.info(f"Total flops after training: {prof.get_total_flops(as_string=False)} or {Decimal(prof.get_total_flops(as_string=False)):.2E}")
logging.info(f"Model params: {Decimal(prof.get_total_params(as_string=False)):.2E}")

prof.end_profile()

model.eval() #Drop gradients for evaluation (increases efficiency)

test_loader = data.DataLoader(test_dataset, shuffle=False)
preds = []
true = []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=False)    

    predictions = np.argmax(outputs[0].cpu().detach().numpy()) # Fetch predicted labels
    labels = batch['labels'].detach().numpy()
    
    true.append(labels)
    preds.append(predictions)
    
f1_score = f1_score(preds, true, average='macro') # Do class distribution invariant computation

logging.info(f"macro averaged F1 score: {f1_score}")

tracker.stop()

end = time.time()
relative_end = perf_counter()
process_end = process_time()
logging.info(f"Elapsed time using time.time: {end - start}")
logging.info(f"Elapsed time using time.perf_counter(): {relative_end - relative_start}")
logging.info(f"Elapsed time using time.process_time(): {process_end}")


