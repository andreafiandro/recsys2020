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
from recSysDataset import BertDataset, CNN_Features_Dataset

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from transformers import BertForSequenceClassification

from RCE import Multi_Label_RCE_Loss
from cnn_model import TEXT_ENSEMBLE, CNN, FEATURES_ENSEMBLE
from nlprecsysutility import RecSysUtility

_PRINT_INTERMEDIATE_LOG = True
# Choose GPU if is available, otherwise cpu
# Pay attention, if you use HPC load the cuda module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create the directory where to save the model checkpoint
os.makedirs(TrainingConfig._checkpoint_path, exist_ok=True)



def train_model(model, dataloaders_dict, datasizes_dict, criterion, optimizer, scheduler, epochs, rce_loss, features=False):

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
    best_rce = math.inf
    best_rce_by_label = None
    saving_file = os.path.join(TrainingConfig._checkpoint_path, 'cnn_model_test.pth')
    #torch.autograd.set_detect_anomaly(True)
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
            eval_logits = eval_targets = None
            # Iterate over data
            for inputs, targets in dataloaders_dict[phase]:
                if features:
                    feats = torch.from_numpy(np.array(inputs[1])).to(device)
                    inputs = torch.from_numpy(np.array(inputs[0])).to(device)
                    
                else:
                    inputs = torch.from_numpy(np.array(inputs)).to(device)              
                targets = targets.to(device)
                optimizer.zero_grad()

                # create computational graph only for training
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # BERT returns a tuple. The first one contains the logits
                    if features:
                        logits, _ = model(inputs, feats)
                    else:
                        logits, _ = model(inputs)

                    # aggiunto per via dell'errore 
                    # Runtime Error: result type Float cannot be cast to the desired output type Long
                    targets = targets.float()
                    loss = criterion(logits, targets).to(device)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:
                        if eval_logits is not None:
                            eval_logits = torch.cat((eval_logits,logits))
                            eval_targets = torch.cat((eval_targets, targets))
                        else:
                            eval_logits = logits
                            eval_targets = targets

                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / datasizes_dict[phase]
            
            
            #output_acc = correct_output_num.double() / datasizes_dict[phase]
            print('{} total loss: {:.4f} '.format(phase,epoch_loss))
            #print('{} output_acc: {:.4f}'.format(phase, output_acc))
            if phase == 'val':
                running_rce = rce_loss(eval_logits, eval_targets)
                print('running rce loss:', running_rce)
                epoch_rce = torch.sum(running_rce)

                if epoch_loss < best_loss:
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
                    
                if epoch_rce < best_rce:
                    print('new best overall rce of {:.4f}'.format(epoch_rce),
                        'improved over previous {:.4f}'.format(best_rce))
                    best_rce = epoch_rce
                    best_rce_by_label = running_rce
                    # print('rce eval loss: {}'.format(epoch_rce))
               
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc (loss): {:4f}'.format(float(best_loss)))
    print('Best val overall rce (loss): {:.4f}'.format(best_rce))
    print('Best val overall rce (loss) by label: ', best_rce_by_label)
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
    classes_support = y_train.astype(bool).sum(axis=0).apply(lambda x: 1 if x == 0 else x )
    test_positive_rates = y_test.astype(bool).sum(axis=0) / len(y_test.index) #positivi/tot
    # classes_support = [1 if x == 0 else x for x in classes_support]
    

    
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

    return [x_train, y_train_reply, y_train_retweet, y_train_retweet_comment, y_train_like], [x_test, y_test_reply, y_test_retweet, y_test_retweet_comment, y_test_like] , classes_support, test_positive_rates

def preprocess_features(df, args):
    df = df.fillna(0)
    x = df[df.columns[:-4]]
    y = df[df.columns[-4:]] # column name prediction

    dummy = RecSysUtility('')
    x = dummy.generate_features_lgb_mod(x, user_features_file=args.ufeatspath) #Note: Sligthly different from other branch this returns text_tokens column
    x = dummy.encode_string_features(x)
    not_useful_cols = ['Tweet_id', 'User_id', 'User_id_engaging']
    x.drop(not_useful_cols, axis=1, inplace=True)
    for col in x.columns[1:]:
        x[col] = x[col].astype(float) #pd.to_numeric(x[col],downcast='float')

    y = y.clip(upper=1)

    # Split in train and test part the chunk
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.testsplit, random_state=42) 
    # Answer to the Ultimate Question of Life, the Universe, and Everything: 42
    
    # Get the support of each classes in the training (used then to balance the loss)
    # classes_support = y_train.value_counts() # single label
    classes_support = y_train.astype(bool).sum(axis=0).apply(lambda x: 1 if x == 0 else x )
    test_positive_rates = y_test.astype(bool).sum(axis=0) / len(y_test.index) #positivi/tot

    return x_train, y_train, x_test, y_test, classes_support, test_positive_rates

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
        default='Text_tokens',
        type=str,
        required=False,
        help="Column name for bert tokens (e.g. \"text tokens\")"
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
    parser.add_argument(
        "--features",
        default=None,
        type=str,
        required=False,
        help="type yes or something <-> to use text+extracted features for cnn"
    )
    parser.add_argument(
        '--ufeatspath',
        default='./checkpoint/user_features_final.csv',
        type=str,
        required=False,
        help='Path to user_features.csv. Default=\'./checkpoint/user_features_final.csv\''
    )
    parser.add_argument(
        '--use_weights',
        default=None,
        type=str,
        required=False,
        help='Use or not weights for loss initialization (yes/no), default=No'
    )
    args = parser.parse_args()
    if args.features:
        print('##### WORKING WITH FEATURES #####')
    not_bert_finetuning = TrainingConfig._not_finetuning_bert

    # Initializing a BERT model
    bert_model = BERT(pretrained=TrainingConfig._pretrained_bert, n_labels=TrainingConfig._num_labels, dropout_prob = TrainingConfig._dropout_prob, freeze_bert = not_bert_finetuning)
    
    # per training incrementali, da mettere meglio nel training config o altrove senza fargli il caricamento del pretrained_bert

    if(not_bert_finetuning):
        checkpoint = torch.load(os.path.join(TrainingConfig._checkpoint_path, 'bert_model_test.pth'))
        bert_model.load_state_dict(checkpoint)
        bert_model.freeze_layers(bert_model.bert)
        bert_model.freeze_layers(bert_model.classifier)

    nrows = None
        
    ##########################################################################
    # Accessing the model configuration
    # if you need to modify these parameters, just create a new configuration:
    # 
    #       from transformers import BertForSequenceClassification, BertConfig
    #       config = BertConfig(... put your parameters here ...)
    #       model = BertForSequenceClassification(config)
    #
    ##########################################################################
    
    df = pd.read_csv(args.data, nrows=nrows) #Put here nrows = ??? for test purposes
    if _PRINT_INTERMEDIATE_LOG:
        print('DATASET SHAPE: '+ str(df.shape))
        print('HEAD FUNCTION: '+ str(df.head()))

    number_of_rows = len(df.index)
    print("Number of rows "+str(number_of_rows))
    if(args.features):
        cnn = CNN(dim=798, length=21) #768 cls +30 features = 768 % 21 => batch_sizex21x38
        model = FEATURES_ENSEMBLE(bert = bert_model, model_b = cnn)
        x_train, y_train, x_test, y_test, classes_support, test_positive_rates = preprocess_features(df, args)
        #CNN_Features_Dataset(x_tokens, x_features, y)
        train_data = CNN_Features_Dataset(x_train.iloc[:, 0].values.tolist(), x_train.iloc[:, 1:].values.tolist(), y_train.values.tolist())
        test_data = CNN_Features_Dataset(x_test.iloc[:, 0].values.tolist(), x_test.iloc[:, 1:].values.tolist(), y_test.values.tolist())
    else:
        cnn = CNN()
        model = TEXT_ENSEMBLE(bert = bert_model, model_b = cnn)
        train_chunk, test_chunk, classes_support, test_positive_rates = preprocessing(df, args)
        # create the dataset objects
        train_data = BertDataset(xy_list=train_chunk)
        test_data = BertDataset(xy_list=test_chunk)

    if _PRINT_INTERMEDIATE_LOG:
        print(model.config)

    number_of_training_rows = len(train_data)
    print("Number of training rows "+str(number_of_training_rows))

    if _PRINT_INTERMEDIATE_LOG:
        print('Different training classes: \n' + str(classes_support))
    # create the dataloaders for the training loop
    dataloaders_dict = {'train': torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers),
                    'val': torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
                   }
    datasizes_dict = {'train':len(train_data),
                    'val':len(test_data)              
                    }
    # move model to device before optimizer instantiation
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    
    # instantiate CrossEntropy with class penalization based on class support

    # sostituita da altro sopra perchè il calcolo dei pesi è cambiato  
    # loss_weights = []
    # for class_support in classes_support:
    #     loss_weights.append(len(train_data) / class_support)
    # check this patch for not seen labels
    # if len(loss_weights) < TrainingConfig._num_labels: 
    #     loss_weights.extend([1] * (TrainingConfig._num_labels - len(loss_weights)))

    # weights as NEGATIVES / POSITIVES Team JP
    # It is a weight of positive examples. Must be a vector with length equal to the number of classes
    # https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss
    positive_rates = torch.FloatTensor(test_positive_rates)
    if args.use_weights is not None:
        loss_weights = classes_support.apply(lambda positives: (number_of_training_rows-positives)/positives).tolist()
        #loss_weights = [nrows/ classes_support[0], nrows/classes_support[1]]
        
        if _PRINT_INTERMEDIATE_LOG:
            print('LOSS WEIGHTS: '+str(loss_weights))
        loss_weights = torch.tensor(loss_weights)
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device)
    else:
        print('### No Loss Weights ###')
        criterion = nn.BCEWithLogitsLoss().to(device) #No weights

    rce_loss = Multi_Label_RCE_Loss(ctr = positive_rates).to(device)
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
    
    train_model(model, dataloaders_dict, datasizes_dict, criterion, optimizer, scheduler, args.epochs, rce_loss, args.features)

if __name__ == "__main__":
    main()