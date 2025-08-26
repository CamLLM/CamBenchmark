import torch                       
import torch.nn.functional  as F    
from torch import Tensor          
from modelscope import AutoTokenizer, AutoModel     
import numpy as np 
import pandas as pd    
from tqdm import tqdm
from tqdm import trange
from sentence_transformers import SentenceTransformer


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def tokenize(tokenizer, input_texts, eod_id, max_length,device):
    batch_dict = tokenizer(input_texts, padding=False, truncation=True, max_length=max_length-2).to(device)
    for seq, att in zip(batch_dict["input_ids"], batch_dict["attention_mask"]):
        seq.append(eod_id)
        att.append(1)
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
    batch_dict = batch_dict.to(device)
    return batch_dict
    

def get_model(model_name,model_path,max_length=None,device=None):
    if model_name in ("bge-large-zh-v1.5","gte-large-zh","gte_Qwen2-1.5B-instruct","gte_Qwen2-7B-instruct","Conan-embedding-v1"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,device_map='auto')
        max_length=max_length
        device = device
        return tokenizer,model,max_length,device
    
    elif model_name == "m3e-large":
        model = SentenceTransformer(model_path)
        model.eval()
        device = device
        return model,device
        
    elif model_name in ("Qwen3-Embedding-4B","Qwen3-Embedding-8B"):
        device = device
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16,device_map="auto")
        max_length = max_length
        return tokenizer,model,max_length,device

def get_embedding(model_name,model_path,max_length,device,queries:list[str],batch_size=10):
    batch_size = batch_size
    if model_name in ("bge-large-zh-v1.5","gte-large-zh"):
        tokenizer,model,max_length,device = get_model(model_name,model_path,max_length,device)
        # 获得向量
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            batch_dict = tokenizer(batch_queries, padding=True, truncation=True, return_tensors='pt',max_length=max_length).to(device)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = outputs[0][:, 0]
                query_embeddings.append(F.normalize(embeddings,  p=2, dim=1))
            del batch_dict, outputs
        query_embeddings = torch.cat(query_embeddings,  dim=0)
        query_embeddings = query_embeddings.cpu().numpy()

        return query_embeddings

    elif model_name in ("gte_Qwen2-1.5B-instruct","gte_Qwen2-7B-instruct","Conan-embedding-v1") :
        tokenizer,model,max_length,device = get_model(model_name,model_path,max_length,device)
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            batch_dict = tokenizer(batch_queries, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state,  batch_dict["attention_mask"])
                query_embeddings.append(F.normalize(embeddings,  p=2, dim=1))
            del batch_dict, outputs
        query_embeddings = torch.cat(query_embeddings,  dim=0)
        query_embeddings = query_embeddings.cpu().numpy()
        return query_embeddings

    elif model_name in ("Qwen3-Embedding-4B","Qwen3-Embedding-8B"):
        tokenizer,model,max_length,device = get_model(model_name,model_path,max_length,device)
        eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            batch_dict = tokenize(tokenizer, batch_queries, eod_id, max_length,device)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state,  batch_dict["attention_mask"])
                # embeddings = embeddings[:, :dim]  # 设置不同嵌入维度
                query_embeddings.append(F.normalize(embeddings,  p=2, dim=1))
            del batch_dict, outputs
        query_embeddings = torch.cat(query_embeddings,  dim=0)
        query_embeddings = query_embeddings.cpu().numpy()
        return query_embeddings

    elif model_name == "m3e-large":
        model,device = get_model(model_name,model_path,max_length,device)
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i+batch_size]
            with torch.no_grad():
                embeddings = model.encode(batch_queries)
                query_embeddings.append(embeddings)
        query_embeddings = np.concatenate(query_embeddings,  axis=0)
        return query_embeddings


