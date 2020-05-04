import argparse
import math
import os
import pdb
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from bert_model import BERT
from config import TrainingConfig
from recSysDataset import BertDataset

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from transformers import BertForSequenceClassification

_PRINT_INTERMEDIATE_LOG = True
# Choose GPU if is available, otherwise cpu
# Pay attention, if you use HPC load the cuda module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create the directory where to save the model checkpoint
os.makedirs(TrainingConfig._checkpoint_path, exist_ok=True)



def train_model(model, dataloaders_dict, datasizes_dict, criterion, optimizer, scheduler, epochs):

    """This function does the training of the model
    
    Arguments:
        model {transformers.BertForSequenceClassification} -- model to train
        dataloaders_dict {dict} -- dict containing dataloaders for keys 'train' and 'val'
        datasizes_dict {dict} -- dict containing dataset sizes for keys 'train' and 'val'
        criterion {torch.nn.CrossEntropyLoss} -- loss criterion for training
        optimizer {torch.optim} -- optimizer for the training
        scheduler {torch.optim.lr_scheduler} -- scheduler for learning rate decay
        epochs {int} -- number of epochs for the training
    """

    since = time.time()
    print('TRAINING STARTED')

    best_loss = math.inf
    saving_file = os.path.join(TrainingConfig._checkpoint_path, 'bert_model_test.pth')
    saving_file_finetuning = os.path.join(TrainingConfig._checkpoint_path, 'bert_model_test_finetuning.pth')

    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            correct_output_num = 0

            # Iterate over data
            for inputs, targets in dataloaders_dict[phase]:
                
                inputs = torch.from_numpy(np.array(inputs)).to(device)              
                targets = targets.to(device)
                optimizer.zero_grad()

                # create computational graph only for training
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # BERT returns a tuple. The first one contains the logits
                    
                    logits, cls_output = model(inputs)

                    # aggiunto per via dell'errore 
                    # Runtime Error: result type Float cannot be cast to the desired output type Long
                    targets = targets.float()

                    loss = criterion(logits, targets)
                    loss.to(device)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                    #statistics
                    running_loss += loss.item() * inputs.size(0)

                    # da modificare per l'accuracy multilabel
                    # correct_output_num += torch.sum(torch.max(logits, 1)[1] == targets)

            epoch_loss = running_loss / datasizes_dict[phase]
            #output_acc = correct_output_num.double() / datasizes_dict[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            #print('{} output_acc: {:.4f}'.format(phase, output_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                model_cpu = model.cpu().state_dict()

                # Print model's state_dict
                # print("Model's state_dict:")
                # for param_tensor in model_cpu:
                #     print(param_tensor, "\t", model_cpu[param_tensor].size())

                torch.save(model_cpu, saving_file)
                print('checkpoint saved in '+saving_file)
                model.to(device)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_loss)))
    
    return model.load_state_dict(torch.load(saving_file))



def preprocessing(df, args):
    """Preprocessing to obtain the train and test data starting from the dataframe
    
    Arguments:
        dataframe {pandas.DataFrame} -- DataFrame of pandas containing the input data
    """

    # Fill the dataset NaN cells with 0
    df = df.fillna(0)

    # Select the columns of our interest:
    #   - text tokens
    #   - tweet type
    #   - Reply engagement timestamp
    #   - Retweet engagement timestamp
    #   - Retweet with comment engagement timestamp
    #   - Like engagement timestamp
    x = df[args.tokcolumn] # Text_tokens
    # y = df[args.predcolumn] # single label
    y = df[df.columns[-4:]] # column name prediction
    
    ##########################################################################
    # The labels are not enum but are represented in the dataset
    # as empty cell if there wasn't no engagment or with a timestamp if was an engagment.
    # The necessary transformations will take place through df.fillna(0) and
    # transformation lambda
    # Set 1 for timestamps (values > 0)
    # clip(upper=threshold) works in that way
    #   
    #   set y[i] = threshold, if y[i] > threshold
    ##########################################################################
    y = y.clip(upper=1)

    # Split in train and test part the chunk
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.testsplit, random_state=42) 
    # Answer to the Ultimate Question of Life, the Universe, and Everything

    # Get the support of each classes in the training (used then to balance the loss)
    # classes_support = y_train.value_counts() # single label
    classes_support = y_train.astype(bool).sum(axis=0)
    classes_support = [1 if x == 0 else x for x in classes_support]

    
    # In this case, due the nature of text tokens field, we will have a list of string. 
    # Each string is a sequence of token ids separed by | , that have to be correctly transformed into a list 
    # (this will be done by text_data function).
    x_train = x_train.values.tolist()
    x_test = x_test.values.tolist()
    
    y_train_reply = y_train['Reply_engagement_timestamp'].values.tolist()
    y_train_retweet = y_train['Retweet_engagement_timestamp'].values.tolist()
    y_train_retweet_comment = y_train['Retweet_with_comment_engagement_timestamp'].values.tolist()
    y_train_like = y_train['Like_engagement_timestamp'].values.tolist()

    y_test_reply = y_test['Reply_engagement_timestamp'].values.tolist()
    y_test_retweet = y_test['Retweet_engagement_timestamp'].values.tolist()
    y_test_retweet_comment = y_test['Retweet_with_comment_engagement_timestamp'].values.tolist()
    y_test_like = y_test['Like_engagement_timestamp'].values.tolist()

    ##########################################################################
    # pandas.get_dummies
    # Convert categorical variable into dummy/indicator variables.
    # Examples
    #
    # s = pd.Series(list('abca'))
    # pd.get_dummies(s)
    #    a  b  c
    # 0  1  0  0
    # 1  0  1  0
    # 2  0  0  1
    # 3  1  0  0
    # pd.get_dummies(s).values.tolist()
    # [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    #
    # ... it is like a one hot encoding for labels
    ##########################################################################

    #y_train = pd.get_dummies(y_train).values.tolist()
    #y_test = pd.get_dummies(y_test).values.tolist()


    return [x_train, y_train_reply, y_train_retweet, y_train_retweet_comment, y_train_like], [x_test, y_test_reply, y_test_retweet, y_test_retweet_comment, y_test_like] , classes_support



def main():
    """Main function for doing sequence classification with BERT

    input example:
    $ python bert_training.py \
            --data isa_puppy_chunk_recsys2020.csv \
            --tokcolumn Text_tokens \
            --predcolumn Reply_engagement_timestamp \
            --epochs 3 \
            --batch 16 \
            --workers 2 \
            --testsplit 0.10            
    """
  
    parser = argparse.ArgumentParser()

    #read user parameters
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--tokcolumn",
        default=None,
        type=str,
        required=True,
        help="Column name for bert tokens (e.g. \"text tokens\")"
    )
    parser.add_argument(
        "--predcolumn",
        default=None,
        type=str,
        required=True,
        help="Column name for prediction (e.g. \"Reply_angagement_timestamp\" etc.)"
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        required=True,
        help="Number of epochs for the training"
    )
    parser.add_argument(
        "--batch",
        default=None,
        type=int,
        required=True,
        help="Batch size for the training"
    )
    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        required=True,
        help="Number of workers for the training"
    )
    parser.add_argument(
        "--testsplit",
        default=None,
        type=float,
        required=True,
        help="Test split size (e.g. 0.10)"
    )
    
    args = parser.parse_args()

    # Initializing a BERT model
    model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = True)
   
    ##########################################################################
    # Accessing the model configuration
    # if you need to modify these parameters, just create a new configuration:
    # 
    #       from transformers import BertForSequenceClassification, BertConfig
    #       config = BertConfig(... put your parameters here ...)
    #       model = BertForSequenceClassification(config)
    #
    ##########################################################################
    if _PRINT_INTERMEDIATE_LOG:
        print(model.config)

    df = pd.read_csv(args.data)
    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(df.shape))
        print('HEAD FUNCTION: '+ str(df.head()))


    train_chunk, test_chunk, classes_support = preprocessing(df, args)

    if _PRINT_INTERMEDIATE_LOG:
        print('Different training classes: \n' + str(classes_support))

    # create the dataset objects
    train_data = BertDataset(xy_list=train_chunk)
    test_data = BertDataset(xy_list=test_chunk)

    # create the dataloaders for the training loop
    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers),
                    'val': torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
                   }
    datasizes_dict = {'train':len(train_data),
                    'val':len(test_data)              
                    }

    

    # move model to device before optimizer instantiation
    model.to(device)

    # create optimizer with different learning rates per layer
    optimizer = optim.Adam(
        [
            {'params': model.bert.parameters(), 'lr': TrainingConfig._bert_lr},
            {'params': model.classifier.parameters(), 'lr': TrainingConfig._cls_lr}
        ]
    )

    # instantiate CrossEntropy with class penalization based on class support
    loss_weights = []


    for class_support in classes_support:
        loss_weights.append(len(train_data) / class_support)

    #check this patch for not seen labels    
    # if len(loss_weights) < TrainingConfig._num_labels: 
    #     loss_weights.extend([1] * (TrainingConfig._num_labels - len(loss_weights)))

    #loss_weights = [nrows/ classes_support[0], nrows/classes_support[1]]
    if _PRINT_INTERMEDIATE_LOG:
        print('LOSS WEIGHTS: '+str(loss_weights))
    loss_weights = torch.tensor(loss_weights)
    
    # criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device) # single label
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device) 

    # def calculate_ctr(gt):
    #    positive = len([x for x in gt if x == 1])
    #    ctr = positive/float(len(gt))
    #    return ctr
    #
    # def compute_rce(pred, gt):
    #    cross_entropy = log_loss(gt, pred)
    #    data_ctr = calculate_ctr(gt)
    #    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    #    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0
    #
    # criterion = compute_rce


    # set scheduler for learning rate decay
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=TrainingConfig._scheduler_step, 
                                    gamma=TrainingConfig._scheduler_gamma)

    train_model(model, dataloaders_dict, datasizes_dict, criterion, optimizer, scheduler, args.epochs)

if __name__ == "__main__":
    main()