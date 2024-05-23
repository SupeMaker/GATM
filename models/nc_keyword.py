from transformers import AutoTokenizer, AutoModel
import torch

#
# class KeyBertModel():
#     def __int__(self, embedding_type='bert-base-uncased', key_model='paraphrase-MiniLM-L6-v2'):
#         self.tokenizer = AutoTokenizer.from_pretrained('D:\\AI\\model\\' + embedding_type)
#         self.model = AutoModel.from_pretrained('D:\\AI\\model\\' + key_model)
#
#     def extract_keyword(self, text):
#         encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
#         return sentence_embeddings
#
#     # Mean Pooling - Take attention mask into account for correct averaging
#     def mean_pooling(model_output, attention_mask):
#         token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

from keybert import KeyBERT


class KeyBertModel():
    # ='bert-base-uncased'
    def __int__(self, embedding_type='bert-base-uncased', key_model='paraphrase-MiniLM-L6-v2', **kwargs):
        self.model = KeyBERT(model='D:\\AI\\model\\' + embedding_type)

    def extract_keyword(self, doc):
        doc_embeddings, word_embeddings = self.model.extract_embeddings(doc)
        return doc_embeddings, word_embeddings