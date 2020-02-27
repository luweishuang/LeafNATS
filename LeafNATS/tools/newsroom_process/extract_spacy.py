import spacy, os
import json, gzip
from multiprocessing import Pool

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])


def process_data(input_):
    article = input_['text']
    summary = input_['summary']
    title = input_['title']
    if article == None or summary == None or title == None:
        return ''
    article = nlp(article)
    summary = nlp(summary)
    title = nlp(title)
    sen_arr = []
    for sen in article.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    article = ' '.join(sen_arr)
    sen_arr = []
    for sen in summary.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    summary = ' '.join(sen_arr)
    sen_arr = []
    for sen in title.sents:
        sen = [k.text for k in sen if '\n' not in k.text]
        sen = ['<s>']+sen+['</s>']
        sen = ' '.join(sen)
        sen_arr.append(sen)
    title = ' '.join(sen_arr)
    sen_arr = [title, summary, article]
    return '<sec>'.join(sen_arr)


def process_curfile(fin_path, fout_path):
    cnt = 0
    batcher = []
    fout = open(fout_path, 'w')
    with gzip.open(fin_path) as fin:
        for ln in fin:
            line = json.loads(ln)
            cnt += 1
            batcher.append(line)
            if len(batcher) == 64:
                pool = Pool(processes=16)
                result = pool.map(process_data, batcher)
                pool.terminate()
                for itm in result:
                    if len(itm) > 1:
                        fout.write(itm+'\n')
                batcher = []

        if len(batcher) > 0:
            pool = Pool(processes=16)
            result = pool.map(process_data, batcher)
            pool.terminate()
            for itm in result:
                if len(itm) > 1:
                    fout.write(itm+'\n')
    fout.close()
    return


current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))  # 获取当前文件的父目录
repo_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../../../")
src_data_dir = os.path.join(repo_dir, 'src_data/newsroom')

fin_path = os.path.join(src_data_dir, 'test.jsonl.gz')
fout_path = 'plain_data/test.txt'
process_curfile(fin_path, fout_path)

fin_path = os.path.join(src_data_dir, 'dev.jsonl.gz')
fout_path = 'plain_data/dev.txt'
process_curfile(fin_path, fout_path)

fin_path = os.path.join(src_data_dir, 'train.jsonl.gz')
fout_path = 'plain_data/train.txt'
process_curfile(fin_path, fout_path)