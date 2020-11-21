from pytorch_transformers.modeling_bert import *
from pytorch_transformers.modeling_roberta import *
from pytorch_transformers.modeling_xlnet import *
import torch

##################################################################################################
############################################# BERT ###############################################
##################################################################################################


class EmbInputBertEmbeddings(BertEmbeddings):
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_ids = input_ids.to(dtype=next(self.parameters()).dtype)
        
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(position_ids)
        
        embeddings = input_ids + self.position_embeddings(position_ids)
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EmbInputBertModel(BertModel):
    def __init__(self, config):
        super(EmbInputBertModel, self).__init__(config)
        self.embeddings = EmbInputBertEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        
        assert input_ids.sum() != 0
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) 
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class EmbInputBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(EmbInputBertForSequenceClassification, self).__init__(config)
        self.bert = EmbInputBertModel(config)
        self.init_weights()


class EmbInputBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super(EmbInputBertForTokenClassification, self).__init__(config)
        self.bert = EmbInputBertModel(config)
        self.init_weights()


class EmbInputBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config):
        super(EmbInputBertForQuestionAnswering, self).__init__(config)
        self.bert = EmbInputBertModel(config)
        self.init_weights()

class EmbInputBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(EmbInputBertForMaskedLM, self).__init__(config)
        self.bert = EmbInputBertModel(config)
        self.init_weights()


class BertEmbLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertEmbLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        return hidden_states

class BertOnlyMEmbLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMEmbLMHead, self).__init__()
        self.predictions = BertEmbLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertForMaskedEmbLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedEmbLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMEmbLMHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]
        transformed = self.cls(sequence_output)
        outputs = (transformed,) + outputs
        return outputs

class EmbInputBertForMaskedEmbLM(BertForMaskedEmbLM):
    def __init__(self, config):
        super(EmbInputBertForMaskedEmbLM, self).__init__(config)
        self.bert = EmbInputBertModel(config)
        self.init_weights()


##################################################################################################
############################################# XLNet ##############################################
##################################################################################################


class EmbInputXLNetModel(XLNetModel):
    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        word_emb_k = input_ids
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        if token_type_ids is not None:
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
            cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)

            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
                                   r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping,
                                   head_mask=head_mask[i])
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)
        
        outputs = (output.permute(1, 0, 2).contiguous(), new_mems)
        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, new_mems, (hidden_states), (attentions)


class EmbInputXLNetForSequenceClassification(XLNetForSequenceClassification):
    def __init__(self, config):
        super(EmbInputXLNetForSequenceClassification, self).__init__(config)
        self.transformer = EmbInputXLNetModel(config)
        self.init_weights()


class EmbInputXLNetForQuestionAnswering(XLNetForQuestionAnswering):
    def __init__(self, config):
        super(EmbInputXLNetForQuestionAnswering, self).__init__(config)
        self.transformer = EmbInputXLNetModel(config)
        self.init_weights()


##################################################################################################
########################################### Roberta ##############################################
##################################################################################################

class EmbInputRobertaEmbeddings(EmbInputBertEmbeddings):
    def __init__(self, config):
        super(EmbInputRobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, 
                    dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        return super(EmbInputRobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids)


class EmbInputRobertaModel(EmbInputBertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    
    def __init__(self, config):
        super(EmbInputRobertaModel, self).__init__(config)
        self.embeddings = EmbInputRobertaEmbeddings(config)
        self.init_weights()


class EmbInputRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super(EmbInputRobertaForSequenceClassification, self).__init__(config)
        self.roberta = EmbInputRobertaModel(config)
        self.init_weights()

class EmbInputRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(EmbInputRobertaForMaskedLM, self).__init__(config)
        self.roberta = EmbInputRobertaModel(config)
        self.init_weights()
