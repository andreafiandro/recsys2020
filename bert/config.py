class TrainingConfig():
    _bert_lr = 5e-5
    _cls_lr = .001
    
    _scheduler_step = 3
    _scheduler_gamma = 0.1

    _dropout_prob = .1
    _text_cnn_dropout_prob = .2
    _n_labels = 4


    _checkpoint_path = 'checkpoint/'
    _pretrained_bert = 'bert-base-multilingual-cased'
