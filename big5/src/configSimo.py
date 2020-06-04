from pytorch_pretrained_bert import BertTokenizer
from bert_serving.server.helper import get_args_parser

START_LINE = 0
CHUNK_NUM = "140"
FILENAME = "../input/training_chunk_"+CHUNK_NUM+".csv"
PAD_MAX_LEN = 192
SEQ_MAX_LEN = 'NONE'
POOL_STR = 'CLS_TOKEN'
PORT = '5555'
PORT_OUT = '5556'
BC_IP = '0.0.0.0'

# PATH_TO_MODEL = "../../BERT/big5/Models/"
PATH_TO_MODEL = "../models/"
PRETRAINED_BERT_PATH = "/home/Venv/Documents/dataset/multi_cased_L-12_H-768_A-12"
PRETRAINED_PYTORCH_BERT = 'bert-base-multilingual-cased'

TOKENIZER = BertTokenizer.from_pretrained(
    'bert-base-multilingual-cased',
    do_lower_case=False
)
ARGS = get_args_parser().parse_args(
        ['-model_dir', PRETRAINED_BERT_PATH,
         '-port', PORT,
         '-port_out', PORT_OUT,
         '-max_seq_len', SEQ_MAX_LEN,
         '-mask_cls_sep',
         '-device_map=0',
         '-pooling_strategy', POOL_STR]
)
