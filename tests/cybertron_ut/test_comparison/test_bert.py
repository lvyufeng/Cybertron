import unittest
import mindspore
import torch
import numpy as np
from cybertron.transformers import BertModel, BertConfig, BertForSequenceClassification
from cybertron.utils import convert_state_dict
from mindspore import Tensor
from transformers import BertConfig as ptBertConfig
from transformers import BertModel as ptBertModel
from transformers import BertForSequenceClassification as ptBertForSequenceClassification

class TestBertComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.ms_config = BertConfig(vocab_size=1000,
                                    hidden_size=256,
                                    num_hidden_layers=4,
                                    num_attention_heads=4,
                                    intermediate_size=128)
        self.pt_config = ptBertConfig(vocab_size=1000,
                                      hidden_size=256,
                                      num_hidden_layers=4,
                                      num_attention_heads=4,
                                      intermediate_size=128)

    def test_bert_comparsion(self):
        model = BertModel(self.ms_config)
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        pt_model = ptBertModel(self.pt_config)
        pt_model.eval()
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        outputs_pt, pooled_pt = pt_model(input_ids=pt_input_ids)

        ms_dict = convert_state_dict(pt_model, 'bert')
        mindspore.load_param_into_net(model, ms_dict)

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        outputs, pooled = model(ms_input_ids)

        assert np.allclose(outputs.asnumpy(), outputs_pt.detach().numpy(), atol=1e-5)
        assert np.allclose(pooled.asnumpy(), pooled_pt.detach().numpy(), atol=1e-5)

    def test_bert_sequence_comparison(self):
        model = BertForSequenceClassification(self.ms_config)
        model.set_train(False)
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [0] * 500

        
        pt_model = ptBertForSequenceClassification(self.pt_config)
        pt_model.eval()

        ms_dict = convert_state_dict(pt_model, 'bert')
        mindspore.load_param_into_net(model, ms_dict)
        model.classifier.weight.set_data(mindspore.Tensor(pt_model.classifier.weight.detach().numpy()))
        model.classifier.bias.set_data(mindspore.Tensor(pt_model.classifier.bias.detach().numpy()))

        ms_input_ids = Tensor(input_ids, mindspore.int32).reshape(1, -1)
        (outputs, ) = model(ms_input_ids)
        pt_input_ids = torch.IntTensor(input_ids).reshape(1, -1)
        (outputs_pt,) = pt_model(input_ids=pt_input_ids)

        assert np.allclose(outputs.asnumpy(), outputs_pt.detach().numpy(), atol=1e-5)
