class TrainingConfig():
    _bert_lr = 5e-5
    _cls_lr = .001
    
    _scheduler_step = 3
    _scheduler_gamma = 0.1

    _dropout_prob = .5

    _num_labels = 4

    _not_finetuning_bert = True
    _checkpoint_path = 'checkpoint/'
    _pretrained_bert = 'bert-base-multilingual-cased'

class TestConfig():
    _output_dir = '../recsys2020_submission'
    
class PreproDatasetConfig():
    _chunk_size = 1000000
    _test_file  = '../recsys2020_test_chunk/test_submission.tsv'
    _output_dir = '../recsys2020_test_chunk'