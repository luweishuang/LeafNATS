import argparse
import torch


parser = argparse.ArgumentParser()

# Use in the framework and cannot remove.
parser.add_argument('--task', default='app', help='app')
parser.add_argument('--data_dir', default='../mt_data_spacy/', help='directory that store the data.')
parser.add_argument('--file_corpus', default='train.txt', help='file store training documents.')
parser.add_argument('--file_val', default='val.txt', help='val data')

parser.add_argument('--n_epoch', type=int, default=35, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=100, help='How often you want to save model?')
parser.add_argument('--val_num_batch', type=int, default=30, help='how many batches')
parser.add_argument('--nbestmodel', type=int, default=10, help='How many models you want to keep?')

parser.add_argument('--continue_training', type=bool, default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=bool, default=False, help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=bool, default=False, help='move average')
parser.add_argument('--use_optimal_model', type=bool, default=True, help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='0,0', help='epoch,batch')
parser.add_argument('--is_lower', type=bool, default=True, help='convert all tokens to lower case?')

# User specified parameters.
# parser.add_argument('--device', default=torch.device("cpu"), help='device')    # "cuda:0"
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
parser.add_argument('--file_vocab', default='vocab', help='file store training vocabulary.')
parser.add_argument('--max_vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_minfreq', type=int, default=5, help='min word frequency')
parser.add_argument('--emb_dim', type=int, default=128, help='source embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=256, help='decoder hidden dimension')
parser.add_argument('--src_seq_lens', type=int, default=1000, help='length of source documents.')   # !!! 有可能被覆盖写
parser.add_argument('--sum_seq_lens', type=int, default=300, help='length of target documents.')   # 限制生成摘要的最长长度
parser.add_argument('--ttl_seq_lens', type=int, default=20, help='length of target documents.')    # 限制生成标题的最长长度

parser.add_argument('--rnn_network', default='lstm', help='gru | lstm')
parser.add_argument('--attn_method', default='luong_concat', help='luong_concat | luong_general')
parser.add_argument('--repetition', default='vanilla', help='vanilla | temporal. Repetition Handling')
parser.add_argument('--pointer_net', type=bool, default=True, help='Use pointer network?')
parser.add_argument('--oov_explicit', type=bool, default=True, help='explicit OOV?')
parser.add_argument('--attn_decoder', type=bool, default=True, help='attention decoder?')
parser.add_argument('--share_emb_weight', type=bool, default=True, help='share_emb_weight')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')
# for beam search
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--beam_size', type=int, default=5, help='beam size.')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for beam search.')
parser.add_argument('--copy_words', type=bool, default=True, help='Do you want to copy words?')
parser.add_argument('--task_key', default='summary', help='summary | title')
# for app
parser.add_argument('--app_model_dir', default='model/', help='directory that stores models.')
parser.add_argument('--app_data_dir', default='test/', help='directory that stores data.')
args = parser.parse_args()


from nats.headline2_summary2_app.model_app import modelApp

model = modelApp(args)

content_in = "villagers , fishermen and hotel residents found the dolphins ' carcasses on friday and alerted officials .it was not immediately clear what killed the 400 dolphins , though scientists ruled out poisoning .narriman jidawi , a marine biologist at the institute of marine science in zanzibar , said their carcasses were strewn along a 4km stretch of nungwi .but the bottleneck dolphins , which live in deep offshore waters , had empty stomachs , meaning that they could have been disoriented and were swimming for some time to reorient themselves .they did not starve to death and were not poisoned , jidawi said .in the united states , experts were investigating the possibility that sonar from us submarines could have been responsible for a similar incident in marathon , florida , where 68 deep-water dolphins stranded themselves in march 2005 .a us navy task force patrols the east africa coast .a navy official was not immediately available for comment , but the service rarely comments on the location of submarines at sea .the deaths are a blow to the tourism industry in zanzibar , where thousands of visitors go to watch and swim with wild dolphins."
model.app2Go()
summary_out = model.app2Go_process(content_in)
print(summary_out)