import torch 
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
import ujson
import os 
import warnings
from huggingface_hub import hf_hub_download
from scaling_retriever.modeling.losses.regulariaztion import init_regularizer, Treshold_reg, Scaled_FLOPS_with_mask, selu
from scaling_retriever.modeling.bidirectional_llama import LlamaBiForMNTP


class LLM2Retriever(torch.nn.Module):
    _tied_weights_keys = None
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # by default, we use NCE loss
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.reg_loss = init_regularizer("FLOPS")
        
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.local_rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.local_rank = 0
        
    def encode(self, **inputs):
        raise NotImplementedError
    
    def gather(self, tensor):
        dtensor = tensor.detach()
        gather_list = [torch.zeros_like(dtensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, dtensor)
        gather_list[self.local_rank] = tensor 
        
        return torch.cat(gather_list, 0)
    
    def forward(self, **inputs):
        # The reps are not normalized to 1: sparse reps obviously not normalized, dense reps tries to normalize but actually not
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0)) # Use dot product to measure similarity.
        rank_loss = self.loss_fn(logits, labels) # Use in-batch negatives, the diagonal logits are the correct logits

        query_reg_loss = self.reg_loss(query_reps)
        doc_reg_loss = self.reg_loss(context_reps)


        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }    
    
    def doc_encode(self, **inputs):
        return self.encode(**inputs)
    
    def query_encode(self, **inputs):
        return self.encode(**inputs)
    
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    @classmethod 
    def build(cls, model_name_or_path, args, resize_len=None, thresh_size=1, config=None):
        if config is not None:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            model_config.update(config)
            print("to modify model config: ", config)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, config=model_config)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        if resize_len is not None:
            base_model.resize_token_embeddings(resize_len) #resize embeddings space if needed
            print(f"resized embeddings space to {resize_len}")
        if args.thresh:
            base_model.add_module('q_thres', Thresholding(method='document',thresh_size=thresh_size))
            base_model.add_module('d_thres',Thresholding(method="document",thresh_size=thresh_size))
            
        if getattr(args, "lora_lm_head_r", None) is not None:
            rank_pattern = {"lm_head": args.lora_lm_head_r}
            target_modules = cls.TARGET_MODULES + ["lm_head"]
        else:
            rank_pattern = None
            target_modules = cls.TARGET_MODULES
        if args.lora:
            #print("*******************")
            lora_config = LoraConfig(
                    base_model_name_or_path=args.model_name_or_path,
                    task_type=None,
                    r=args.lora_r,
                    rank_pattern=rank_pattern,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    inference_mode=False,
                    modules_to_save=args.lora_modules_to_save
                )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(lora_model) 
            lora_model.print_trainable_parameters()
        else:
            model = cls(base_model)

        return model
    
    @classmethod 
    def load(cls, 
             model_name_or_path,
             load_threshold=False,
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False,
             resize_len=None,
             lora_lm_head_r=None,
             thresh_size=1):
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        #print(lora_name_or_path,flush=True)
        # the checkpoint's config.json carries retriever-specific keys
        # (load_threshold, thresh_size, use_bow, ...); resolve it from a local
        # directory or from the huggingface hub
        config_json_path = None
        if lora_name_or_path is not None:
            if os.path.isdir(lora_name_or_path):
                if os.path.exists(os.path.join(lora_name_or_path, "config.json")):
                    config_json_path = os.path.join(lora_name_or_path, "config.json")
            else:
                try:
                    config_json_path = hf_hub_download(lora_name_or_path, "config.json")
                except Exception:
                    config_json_path = None
        if config_json_path is not None:
            with open(config_json_path) as fin:
                config = ujson.load(fin)
                model_config.update(config)
            # checkpoints trained before the bm25->bow rename store the old key names
            for legacy_key, new_key in [("use_bm25", "use_bow"),
                                        ("bm25_doc_scale", "bow_doc_scale"),
                                        ("bm25_query_scale", "bow_query_scale")]:
                if hasattr(model_config, legacy_key) and not hasattr(model_config, new_key):
                    model_config.update({new_key: getattr(model_config, legacy_key)})
        attn_impl = "flash_attention_2" if is_trainable else "sdpa"
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path,config=model_config,ignore_mismatched_sizes=True,attn_implementation=attn_impl) #This model is base_model_path
        modules_to_save = []
        if load_threshold:
            base_model.config.update({'load_threshold':True,'thresh_size':thresh_size})
        elif not hasattr(base_model.config,'load_threshold'):
            base_model.config.update({'load_threshold':False, 'thresh_size':None})
        
        if resize_len is not None and resize_len != base_model.config.vocab_size:
            base_model.resize_token_embeddings(resize_len) #resize embeddings space if needed
            base_model.config.update({'vocab_size':resize_len})
            
        if base_model.config.load_threshold:
            base_model.add_module('q_thres', Thresholding(method='document',thresh_size=base_model.config.thresh_size,grad=is_trainable))
            base_model.add_module('d_thres', Thresholding(method='document',thresh_size=base_model.config.thresh_size,grad=is_trainable))
            modules_to_save += ['q_thres','d_thres']
            print("adding threshold to base model")
        
        print(f"modules to save: {modules_to_save}")
        if lora_name_or_path:
            print("load from lora")
            lora_config = LoraConfig.from_pretrained(lora_name_or_path)
            if len(modules_to_save) > 0:
                lora_config.modules_to_save =  modules_to_save
            if lora_lm_head_r != None:
                lora_config.rank_pattern["lm_head"] = lora_lm_head_r
                lora_config.target_modules.add("lm_head")
            lora_model = PeftModel.from_pretrained(base_model, 
                                                   lora_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=is_trainable)
            if merge_peft:
                print("merge peft")
                print("*************************************************")
                lora_model = lora_model.merge_and_unload()
            else:
                print("in LLM2Retriever:load(): lora_model.print_trainable_parameters()")
                lora_model.print_trainable_parameters()
                print("*************************************************")
            model = cls(lora_model)
        else:
            print("lora not loaded")
            model = cls(base_model)
        print(model.base_model.config)
        return model

    @classmethod
    def load_from_lora(cls,
                       lora_name_or_path,
                       merge_peft=True, 
                       is_trainable=False,
                       load_threshold=False,
                       resize_len=None,
                       thresh_size=1):
        if os.path.isdir(lora_name_or_path):
            adapter_config_path = os.path.join(lora_name_or_path, "adapter_config.json")
        else:
            adapter_config_path = hf_hub_download(lora_name_or_path, "adapter_config.json")
            
        with open(adapter_config_path, "r") as f:
            adapter_config = ujson.load(f)
            
        base_model_name_or_path = adapter_config["base_model_name_or_path"]

        return cls.load(base_model_name_or_path, 
                        lora_name_or_path=lora_name_or_path, 
                        merge_peft=merge_peft, 
                        is_trainable=is_trainable,
                        load_threshold=load_threshold,
                        resize_len=resize_len,
                        thresh_size=thresh_size)
            
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)
   
        
class DecoderOnlyBiSparse(LLM2Retriever):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.vocab_size = self.base_model.config.vocab_size
        if hasattr(self.base_model, "d_thres"):
            print("document thresholding enabled")
        if hasattr(self.base_model, "q_thres"):
            print("query thresholding enabled")
        
    def rerank_forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"])
        doc_reps = self.encode(**inputs["tokenized_docs"])
        logits = (query_reps * doc_reps).sum(dim=-1)
        return logits
    
    def encode(self, **inputs):
        # print(inputs)
        seq_reps = self.base_model(**inputs, return_dict=True).logits #[bz, seq_length, dim]
        seq_reps *= self.base_model.config.hidden_size**-0.25
        attention_mask = inputs["attention_mask"]
        #seq_reps = 0.01 * torch.nn.functional.softplus(100 * seq_reps)
        #reps, _ = torch.max(torch.log(1 + seq_reps) * inputs["attention_mask"].unsqueeze(-1), dim=1) #[bz, vocab_size] 
        
        ## we try efficient encode to see whether it can save memory 
        if self.training:
            reps = torch.log(selu(torch.max(seq_reps + ( 1 - attention_mask.unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
        else:
            reps = torch.log(torch.relu(torch.max(seq_reps + ( 1 - attention_mask.unsqueeze(-1)) * -1e6, dim=1)[0]) + 1)
        
        '''
        if hasattr(self, "sparsity_limit") and self.sparsity_limit != None:
            topk_vals, topk_idx = reps.topk(self.sparsity_limit, dim=1)
            mask = torch.zeros_like(reps)
            mask.scatter_(1, topk_idx, 1.0)
            reps = reps * mask
        '''
        ret_reps = [reps]
        if hasattr(self.base_model.config,"use_bow") and self.base_model.config.use_bow and self.training:
            bow_mask = torch.zeros((len(inputs['input_ids']),self.base_model.config.vocab_size),device=inputs['input_ids'].device,dtype=torch.bool)
            bow_mask.scatter_(1, inputs['input_ids'], True)
            if self.base_model.config.pad_token_id is not None:
                bow_mask[:,self.base_model.config.pad_token_id] = False
            if self.base_model.config.bos_token_id is not None:
                bow_mask[:,self.base_model.config.bos_token_id] = False
            if self.base_model.config.eos_token_id is not None:
                bow_mask[:,self.base_model.config.eos_token_id] = False
            ret_reps.append(bow_mask)
        return torch.stack(ret_reps).squeeze()
    
    def doc_encode(self, **inputs):
        '''
        only for inference
        '''
        seq_reps = self.encode(**inputs)
        assert(self.training == False)
        if hasattr(self.base_model.config, "load_threshold") and self.base_model.config.load_threshold:
            seq_reps = self.base_model.d_thres(seq_reps)
        return seq_reps

    def query_encode(self, **inputs):
        assert(self.training is False)
        reps = self.encode(**inputs)
        if hasattr(self.base_model.config, "load_threshold") and self.base_model.config.load_threshold:
            #print(f"ss {self.base_model.q_thres.thresh}")
            reps = self.base_model.q_thres(reps)
        return reps


class LlamaBiSparse(DecoderOnlyBiSparse):
    TRANSFORMER_CLS = LlamaBiForMNTP
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]

    


class LlamaBiSparseForMarginMSE(LlamaBiSparse):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.MSELoss()
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"]) # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (query_rep * neg_doc_rep).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss(student_margin, teacher_margin)
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = (self.reg_loss(pos_doc_rep) + self.reg_loss(neg_doc_rep)) / 2.

        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }

class LlamaBiSparseForNCE_KLDiv_AdaSparse(LlamaBiSparse):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.nce_loss = torch.nn.CrossEntropyLoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_loss = Scaled_FLOPS_with_mask()
        self.threshold_loss = Treshold_reg()

    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        #print(query_reps.size())
        query_bow_mask = None
        context_bow_mask = None
        if self.base_model.config.use_bow and self.training:
            # if bow mask is enabled, encode will return a size of 3 vector
            context_reps, context_bow_mask = context_reps[:-1].squeeze(), context_reps[-1]
            query_reps, query_bow_mask = query_reps[:-1].squeeze(), query_reps[-1]

        if self.base_model.config.load_threshold:
            context_reps = self.base_model.d_thres(context_reps)
            query_reps = self.base_model.q_thres(query_reps)

        labels = inputs["target_labels"] #[n_query]
        teacher_scores = inputs["teacher_scores"] # [n_query, 1 + num_negs]
        teacher_idxes = inputs["teacher_idxes"] # [n_query, 1 + num_negs]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            if self.base_model.config.use_bow:
                query_bow_mask = self.gather(query_bow_mask).to(torch.bool)
                context_bow_mask = self.gather(context_bow_mask).to(torch.bool)
            # the following logits is cross device.
            # The kl_div loss only consider the logits in the same device
            # as teacher_scores is in the same device
            teacher_idxes = teacher_idxes.view(-1) + self.local_rank * n_context 
            
            # original `query_idexes` is wrong
            # query_idxes = torch.LongTensor([self.local_rank] * n_context).to(labels.device)
            query_idxes = torch.LongTensor(torch.repeat_interleave(torch.arange(n_query), n_context // n_query)) + \
                            self.local_rank * n_query
            query_idxes = query_idxes.to(labels.device)
        else:
            teacher_idxes = teacher_idxes.view(-1)
            query_idxes = torch.repeat_interleave(torch.arange(n_query), n_context // n_query).to(labels.device)

        q_lengths = torch.sum(query_bow_mask,dim=-1).squeeze() if query_bow_mask is not None else -1
        doc_lengths = torch.sum(context_bow_mask,dim=-1).squeeze() if context_bow_mask is not None else -1
        if self.base_model.config.use_bow and self.training:
            # STop gating (paper Formula 4): G = sigmoid((pivot - w_j - beta * I(t_j in orig)) / gamma)
            # `lexical_preserve` is the lexical provenance bias beta; the hardcoded 25 = 1/gamma (gamma = 0.04)
            sorted_vals = context_reps.sort(dim=-1, descending=True).values
            lengths = doc_lengths * self.base_model.config.bow_doc_scale
            lengths = torch.clamp(lengths, min=1, max=context_reps.size()[1] - 1)
            kth_max = sorted_vals[torch.arange(context_reps.size()[0]),lengths-1].unsqueeze(-1)
            knext_max = sorted_vals[torch.arange(context_reps.size()[0]),lengths].unsqueeze(-1)
            context_bow_mask = torch.sigmoid(((kth_max+knext_max)/2 - context_reps - context_bow_mask * self.base_model.config.lexical_preserve) * 25)
            bow_loss_d = (context_bow_mask * context_reps ** 2).sum(dim=-1).mean()

            sorted_vals = query_reps.sort(dim=-1, descending=True).values
            lengths = q_lengths * self.base_model.config.bow_query_scale
            lengths = torch.clamp(lengths, min=1, max=query_reps.size()[1] - 1)
            kth_max = sorted_vals[torch.arange(query_reps.size()[0]),lengths-1].unsqueeze(-1)
            knext_max = sorted_vals[torch.arange(query_reps.size()[0]),lengths].unsqueeze(-1)
            query_bow_mask = torch.sigmoid(((kth_max+knext_max)/2 - query_reps - query_bow_mask * self.base_model.config.lexical_preserve) * 25)
            bow_loss_q = (query_bow_mask * query_reps ** 2).sum(dim=-1).mean()
        else:
            bow_loss_q = 0
            bow_loss_d = 0
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        #print(self.p.grad)
        nce_loss = self.nce_loss(logits, labels)
        kl_logits = logits[query_idxes, teacher_idxes].view(teacher_scores.size())
        #print(logits.size(),kl_logits.size(),teacher_scores.size(),flush=True)
        #print(query_idxes,teacher_idxes,flush=True)
        log_probs = torch.nn.functional.log_softmax(kl_logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        kl_loss = self.kldiv_loss(log_probs, teacher_log_probs)
        
        rank_loss = (nce_loss + kl_loss) / 2.
            
        query_reg_loss = self.reg_loss(query_reps,q_lengths)
        doc_reg_loss = self.reg_loss(context_reps,doc_lengths)
        if self.base_model.config.load_threshold:
            threshold_loss = self.threshold_loss(self.base_model.q_thres.thresh) + self.threshold_loss(self.base_model.d_thres.thresh)
        else:
            threshold_loss = torch.tensor(0,device=rank_loss.device).detach() #trainer will call detach on this, so wrap it in tensor
        # rank_loss is used for backward 
        # we also inspect kldiv and nce losses.
        return {
            "rank": rank_loss, 
            "query_reg": query_reg_loss + bow_loss_q,
            "doc_reg": doc_reg_loss + bow_loss_d,
            "threshold_loss": threshold_loss
        }
    


class LlamaBiSparseForNCE_KLDiv(LlamaBiSparse):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.nce_loss = torch.nn.CrossEntropyLoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_loss = init_regularizer("FLOPS")
        
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        teacher_scores = inputs["teacher_scores"] # [n_query, 1 + num_negs]
        teacher_idxes = inputs["teacher_idxes"] # [n_query, 1 + num_negs]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
            # the following logits is cross device.
            # The kl_div loss only consider the logits in the same device
            # as teacher_scores is in the same device
            teacher_idxes = teacher_idxes.view(-1) + self.local_rank * n_context 
            
            # original `query_idexes` is wrong
            # query_idxes = torch.LongTensor([self.local_rank] * n_context).to(labels.device)
            query_idxes = torch.LongTensor(torch.repeat_interleave(torch.arange(n_query), n_context // n_query)) + \
                            self.local_rank * n_query
            query_idxes = query_idxes.to(labels.device)
        
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        nce_loss = self.loss_fn(logits, labels)
        
        kl_logits = logits[query_idxes, teacher_idxes].view(teacher_scores.size())
        log_probs = torch.nn.functional.log_softmax(kl_logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        kl_loss = self.kldiv_loss(log_probs, teacher_log_probs)
        
        rank_loss = (nce_loss + kl_loss) / 2.
        
        query_reg_loss = self.reg_loss(query_reps)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        # rank_loss is used for backward 
        # we also inspect kldiv and nce losses.
        return {
            "rank": rank_loss, 
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }
        
        
class LlamaBiSparseForKLDiv(LlamaBiSparse):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.rank_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_loss = init_regularizer("FLOPS")
        
    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_queries"])
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        teacher_scores = inputs["teacher_scores"] #[bz, 1 + num_negs]
        
        context_reps = context_reps.view(teacher_scores.size(0),
                                         teacher_scores.size(1),
                                         context_reps.size(-1))
        logits = (query_rep.unsqueeze(1) * context_reps).sum(dim=-1) #[bz, 1 + num_negs] 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        teacher_log_probs = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        rank_loss = self.rank_loss(log_probs, teacher_log_probs)
        
        query_reg_loss = self.reg_loss(query_rep)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        return {
            "rank": rank_loss, 
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }


class Thresholding(torch.nn.Module):
    def __init__(self,method:str="query",val:float=0., k=25, grad=True,thresh_size:int=1) -> None:
        '''
        query use soft thresholding, document use hard
        '''
        super().__init__()
        self.thresh = torch.nn.Parameter(torch.Tensor([val]*thresh_size),requires_grad=grad)
        assert(method in ["query","document"])
        self.method = method
        self.k = k
    def forward(self, reps, thresh=None):
        '''
        this is used during training
        '''
        if thresh == None:
            thresh = self.thresh
        if self.training:
            if self.training is False:
                warnings.warn("Thresholding should only be used during training, but found in evaluation state")
            if self.method == "query":
                return torch.relu(reps-thresh)
            else:
                #return reps / 2.0 * (torch.erf((reps - self.thresh) / 0.1) - torch.erf((reps + self.thresh) / 0.1) + 2)
                return reps * torch.sigmoid(self.k*(reps - thresh))
        else:
            if self.method == "query":
                return torch.relu(reps - thresh)
            else:
                #print(torch.sum(reps>thresh)/reps.size()[0])
                #print(torch.sum((reps-0.001)>thresh)/reps.size()[0])
                reps = reps * (reps > thresh)
                #print(torch.count_nonzero(reps)/reps.size()[0])
                return reps
    
    def use(self, reps, thresh=None):
        if thresh == None:
            thresh = self.thresh
        '''
        this is used for inference stage
        '''
        if self.training is True:
            warnings.warn("Thresholding apply method should only be used during evaluation, but found in training state")
        if self.method == "query":
            return torch.relu(reps - thresh)
        else:
            #print(torch.sum(reps>thresh)/reps.size()[0])
            #print(torch.sum((reps-0.001)>thresh)/reps.size()[0])
            reps = reps * (reps > thresh)
            #print(torch.count_nonzero(reps)/reps.size()[0])
            return reps
        
        return torch.sigmoid(logits / tau)