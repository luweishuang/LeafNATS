import json
import spacy
import glob
import numpy as np
from LeafNATS.data.summarization.load_multitask import *
from LeafNATS.modules.decoder.nats_decoder_pointer_generator import PointerGeneratorDecoder
from LeafNATS.modules.decoding.word_copy import word_copy
from LeafNATS.modules.embedding.nats_embedding import natsEmbedding
from LeafNATS.modules.encoder2decoder.nats_encoder2decoder import natsEncoder2Decoder
from LeafNATS.modules.encoder.nats_encoder_rnn import natsEncoder
from nats.pointer_generator_network.model import modelPointerGenerator


from .beam_search import fast_beam_search
nlp = spacy.load('en_core_web_sm', disable=['logging', 'ner'])


class modelApp(modelPointerGenerator):
    def __init__(self, args):
        super(modelApp, self).__init__(args=args)
        self.args.data_dir = self.args.app_model_dir

    def build_models(self):
        '''
        build all models.
        in this model source and target share embeddings
        '''
        self.base_models['embedding_base'] = natsEmbedding(
            vocab_size=self.batch_data['vocab_size'],
            emb_dim=self.args.emb_dim,
            share_emb_weight=self.args.share_emb_weight
        ).to(self.args.device)

        self.base_models['encoder_base'] = natsEncoder(
            emb_dim=self.args.emb_dim,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['encoder_newsroom_title'] = natsEncoder(
            emb_dim=self.args.src_hidden_dim*2,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['encoder_bytecup_title'] = natsEncoder(
            emb_dim=self.args.src_hidden_dim*2,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['encoder_newsroom_summary'] = natsEncoder(
            emb_dim=self.args.src_hidden_dim*2,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['encoder_cnndm_summary'] = natsEncoder(
            emb_dim=self.args.src_hidden_dim*2,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['encoder2decoder_newsroom_title'] = natsEncoder2Decoder(
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            rnn_network=self.args.rnn_network
        ).to(self.args.device)

        self.base_models['encoder2decoder_bytecup_title'] = natsEncoder2Decoder(
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            rnn_network=self.args.rnn_network
        ).to(self.args.device)

        self.base_models['encoder2decoder_newsroom_summary'] = natsEncoder2Decoder(
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            rnn_network=self.args.rnn_network
        ).to(self.args.device)

        self.base_models['encoder2decoder_cnndm_summary'] = natsEncoder2Decoder(
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            rnn_network=self.args.rnn_network
        ).to(self.args.device)

        self.base_models['pgdecoder_newsroom_title'] = PointerGeneratorDecoder(
            input_size=self.args.emb_dim,
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            attn_method=self.args.attn_method,
            repetition=self.args.repetition,
            pointer_net=self.args.pointer_net,
            attn_decoder=self.args.attn_decoder,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['pgdecoder_bytecup_title'] = PointerGeneratorDecoder(
            input_size=self.args.emb_dim,
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            attn_method=self.args.attn_method,
            repetition=self.args.repetition,
            pointer_net=self.args.pointer_net,
            attn_decoder=self.args.attn_decoder,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['pgdecoder_newsroom_summary'] = PointerGeneratorDecoder(
            input_size=self.args.emb_dim,
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            attn_method=self.args.attn_method,
            repetition=self.args.repetition,
            pointer_net=self.args.pointer_net,
            attn_decoder=self.args.attn_decoder,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.base_models['pgdecoder_cnndm_summary'] = PointerGeneratorDecoder(
            input_size=self.args.emb_dim,
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            attn_method=self.args.attn_method,
            repetition="asee",
            pointer_net=self.args.pointer_net,
            attn_decoder=self.args.attn_decoder,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        # decoder to vocab
        self.base_models['decoder2proj_newsroom_title'] = torch.nn.Linear(
            self.args.trg_hidden_dim, self.args.emb_dim, bias=False).to(self.args.device)
        self.base_models['decoder2proj_bytecup_title'] = torch.nn.Linear(
            self.args.trg_hidden_dim, self.args.emb_dim, bias=False).to(self.args.device)
        self.base_models['decoder2proj_newsroom_summary'] = torch.nn.Linear(
            self.args.trg_hidden_dim, self.args.emb_dim, bias=False).to(self.args.device)
        self.base_models['decoder2proj_cnndm_summary'] = torch.nn.Linear(
            self.args.trg_hidden_dim, self.args.emb_dim, bias=False).to(self.args.device)

    def init_base_model_params(self):
        '''
        Initialize Model Parameters
        '''
        for model_name in self.base_models:
            fl_ = os.path.join(self.args.app_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(torch.load(fl_, map_location=lambda storage, loc: storage))

    def attnWeight2rgbPercent(self, input_):
        maxV = np.max(input_)
        minV = np.min(input_)
        output_ = (input_ - minV) / (maxV - minV)
        return output_

    def app_worker(self):
        '''
        For the beam search in application.
        '''
        with torch.no_grad():
            files_ = glob.glob(os.path.join(self.args.app_data_dir, '*_in.json'))
            for curr_file in files_:
                print("Read {}.".format(curr_file))
                fTmp = re.split('\_', curr_file)[0]
                data_output = {}
                with open(curr_file, 'r') as fp:
                    data_input = json.load(fp)
                data_output['content'] = data_input['content']
                article = nlp(data_input['content'].lower())
                article = ' '.join([wd.text for wd in article])
                article = re.split('\s', article)
                article = list(filter(None, article))
                data_input['content_token'] = article
                self.args.src_seq_lens = len(article)

                ext_id2oov, src_var, src_var_ex, src_arr, src_msk = process_data_app(data_input, self.batch_data['vocab2id'], self.args.src_seq_lens)
                self.batch_data['ext_id2oov'] = ext_id2oov
                src_msk = src_msk.to(self.args.device)

                curr_batch_size = src_var.size(0)
                src_text_rep = src_var.unsqueeze(1).clone().repeat(
                    1, self.args.beam_size, 1).view(-1, src_var.size(1)).to(self.args.device)
                if self.args.oov_explicit:
                    src_text_rep_ex = src_var_ex.unsqueeze(1).clone().repeat(
                        1, self.args.beam_size, 1).view(-1, src_var_ex.size(1)).to(self.args.device)
                else:
                    src_text_rep_ex = src_text_rep.clone()

                def inner_func(cur_task_key):
                    beam_seq, beam_prb, beam_attn_ = fast_beam_search(
                        self.args, self.base_models, self.batch_data,
                        src_text_rep, src_text_rep_ex, curr_batch_size, cur_task_key)
                    beam_out = beam_attn_[:, :, 0].squeeze()[:, :self.args.src_seq_lens].data.cpu().numpy()
                    beam_out = self.attnWeight2rgbPercent(beam_out)
                    trg_arr = word_copy(
                        self.args, beam_seq, beam_attn_, src_msk, src_arr, curr_batch_size,
                        self.batch_data['id2vocab'], self.batch_data['ext_id2oov'])
                    trg_arr = re.split('\s', trg_arr[0])
                    out_arr = []
                    for idx, wd in enumerate(trg_arr):
                        if wd == '<stop>':
                            break
                        if wd != '<s>' and wd != '</s>':
                            # out_arr.append({"key": wd, "attention": beam_out[idx].tolist()})
                            out_arr.append(wd)
                    data_input[cur_task_key] = out_arr
                    data_output[cur_task_key] = ' '.join(out_arr)

                self.args.task_key = 'newsroom_title'
                inner_func(self.args.task_key)
                self.args.task_key = 'newsroom_summary'
                inner_func(self.args.task_key)
                self.args.task_key = 'cnndm_summary'
                inner_func(self.args.task_key)
                self.args.task_key = 'bytecup_title'
                inner_func(self.args.task_key)

                print('Write {}.'.format(fTmp+'_out.json'))
                with open(fTmp+'_out.json', 'w') as fout:
                    json.dump(data_output, fout)


