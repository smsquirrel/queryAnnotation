from typing import List, Union, Optional
import torch
import torch.nn as nn

class Model(nn.Module):   
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.args = args
        self.output = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768))
      
    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            if self.args.model_type == "graphcodebert":
                nodes_mask=position_idx.eq(0)
                token_mask=position_idx.ge(2)        
                inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
                nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
                nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]  
                avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
                inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
                outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
            elif self.args.model_type == "starencoder":
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(49152))[1]
            else:
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
        else:
            if self.args.model_type == "starencoder":
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(49152))[1]
            else: 
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
        return outputs
    

def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Temporal ooling of sequences of vectors and projection onto the unit sphere.

    Args:
        features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
        attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
        return_norms (bool, optional): Whether to additionally return the norms. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors with shape [B, F].
    """

    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings
    
def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu