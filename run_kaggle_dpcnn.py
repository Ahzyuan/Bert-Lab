# coding=utf-8
from main import main
import pandas as pd
import os



if __name__ == "__main__":

    model_name = 'BertDPCNN'
    
    data_dir = '/data/hzy/Bert-Lab/Data/origin'
    output_dir = ".Result/" 
    cache_dir = ".Cache/"
    log_dir = ".Logs/" 
    train_val_rate='8:2'

    # bert-base
    bert_vocab_file = '/data/hzy/Bert-Lab/Bert_weight/bert-base-uncased-vocab.txt'
    bert_model_dir = '/data/hzy/Bert-Lab/Bert_weight/bert-base-uncased.tar.gz'

    # # bert-large
    # bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
    # bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"
    
    train_val_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    all_data = pd.concat([train_val_data, test_data])
    max_length = all_data['Sentence'].apply(lambda x: len(x.split())).max()
    label_list = train_val_data['Category'].drop_duplicates().tolist()
    while (max_length-2)%2!=1:
        max_length+=1
    #print(f'max: {max_length}\ncls_list: {label_list}')
    
    #train_val_nums = len(train_val_data)
    #train_rate = int(train_val_rate.split(":")[0])/10
    #train_data = train_val_data.iloc[:int(train_val_nums*train_rate)]
    #val_data = train_val_data.iloc[int(train_val_nums*train_rate):]

    #test_data['Label']=label_list[0]
    #test_data.drop('Id',axis=1,inplace=True)
    
    #train_data.to_csv(os.path.join(os.path.dirname(data_dir),'train.csv'),index=False)
    #val_data.to_csv(os.path.join(os.path.dirname(data_dir),'dev.csv'),index=False)
    #test_data.to_csv(os.path.join(os.path.dirname(data_dir),'test.csv'),index=False)

    data_dir = os.path.dirname(data_dir)

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == 'BertLSTM':
        from BertLSTM import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args
    
    elif model_name == "BertDPCNN":
        from BertDPCNN import args

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir, max_seq_length=max_length)

    main(config, config.save_name, label_list)
        

