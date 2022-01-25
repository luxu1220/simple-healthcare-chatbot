import logging
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AlbertModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(312, 256, k) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 3, 128)
        self.fc2 = nn.Linear(128, 13)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = x.squeeze()
        x = x.permute(0, 2, 1)
        out = torch.cat([self.conv_and_pool(x, conv)
                        for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


CLASS_MAP = {
    '0': "定义",
    '1': "病因",
    '2': "预防",
    '3': "临床表现(病症表现)",
    '4': "相关病症",
    '5': "治疗方法",
    '6': "所属科室",
    '7': "传染性",
    '8': "治愈率",
    '9': "禁忌",
    '10': "治疗时间",
    '11': "化验/体检方案",
    '12': "其他",
}

RE_RULE = re.compile("[^\u4e00-\u9fa5]")


class ClassifierHandler(BaseHandler):
    """
    判断输入的文本的类别
    """

    def __init__(self):
        super(ClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_path = properties.get("model_dir") + '/model.pt'
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        self.albert_model = AlbertModel.from_pretrained('voidful/albert_chinese_tiny')
        # 初始化模型
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        logger.debug(
            'Model from path {0} loaded successfully'.format(model_path))
        self.mapping = CLASS_MAP
        self.initialized = True

    @staticmethod
    def padding_text(text):
        if len(text) > 230:
            text = text[:230]
        else:
            text += '[PAD]' * (230 - len(text))
        return text

    def preprocess(self, data):
        """ preprocessing.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        text = text.decode('utf-8')
        logger.info("Received text: '%s'", text)

        text = RE_RULE.sub('', text)
        text = self.padding_text(text)

        inputs = (self.albert_model(torch.tensor(self.tokenizer.encode(
            text, add_special_tokens=False)).unsqueeze(0)))['last_hidden_state'].detach().numpy()
        inputs = torch.FloatTensor(inputs).to(self.device)
        return inputs

    def inference(self, inputs):
        prediction = self.model(inputs)[0].argmax().item()
        logger.info("Model predicted: '%s'", prediction)
        prediction = self.mapping[str(prediction)]
        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = ClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e


if __name__ == '__main__':
    inputs = _service.preprocess('糖尿病的主要症状有哪些呢？')

    output = _service.inference(inputs)
    print(output)
