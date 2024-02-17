from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, linear_probe=False, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.lm = lm
        self.transformer = lm.transformer
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            lm.lm_head.weight.dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def embed(self, input_ids:torch.LongTensor):
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        emd = self.transformer.drop(tok_emb + pos_emb) 
        return emd


    def forward(self, input_ids: torch.LongTensor, embed=None):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        # input_lens = (input_ids != 0).sum(dim=-1)
        # transformer_outputs = self.transformer(input_ids)
        # hidden_states = torch.stack(
        #     [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        # )
        # self.score.to(hidden_states.device)
        # if self.linear_probe:
        #     hidden_states = hidden_states.detach()
        # logits = self.score(hidden_states)
        # return logits
        if embed is None:
            embed = self.embed(input_ids)
        
        x  = embed

        for block in self.transformer.h:
            x = block(x)[0]
        transformer_outputs = self.transformer.ln_f(x)
        input_lens = (input_ids != 0).sum(dim=-1)
        hidden_states = torch.stack(
            [transformer_outputs[i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )
        self.score.to(hidden_states.device)
        logits = self.score(hidden_states)
        return logits, embed
    
