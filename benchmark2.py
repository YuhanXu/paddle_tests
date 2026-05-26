import time
import os
os.environ["FLAGS_prim_all"] = "true"
import numpy as np
import paddle
from paddle import nn
# import paddleclas
# import ppdet
# import paddleseg
# import paddleocr.tools.program
# import paddleocr.ppocr.modeling.architectures
# import paddleocr.ppocr.utils.save_load
import paddlenlp
# import ppsci

cinn_denied_ops = [
    # "arg_max",
    # "argmax",
    # "bitwise_and",
    # "concat",
    # "cumsum",
    # "gather",
    # "gather_nd",
    # "lookup_table_v2",
    # "reduce_sum",
    # "sum",          # PIR name for reduce_sum in Paddle 3.x
    # "reduce_max",
    # "max",          # PIR name for reduce_max in Paddle 3.x
    # "slice",
    # "strided_slice",
    # "roll",
    # "tile",
    # "transpose2",
    # "transpose",    # PIR name for transpose2 in Paddle 3.x
    # "range",
    # "arange",
    # "fill_constant",
]

paddle.set_flags({
    "FLAGS_print_ir": False,
    "FLAGS_deny_cinn_ops": ";".join(cinn_denied_ops),
})


def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )

def benchmark(net, input, repeat=10, warmup=3):
    # warm up
    for i in range(warmup):
        net(input)
    paddle.device.synchronize()
    # time
    t = []
    for i in range(repeat):
        t1 = time.time()
        net(input)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("--[benchmark] Run for %d times, the average latency is: %f ms" % (repeat, np.mean(t)))

class TestBase:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.input = self.init_input()
        self.net = None
        self.cinn_net = None

    def init_model(self):
        raise NotImplementedError

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 224, 224])

    def get_net(self, use_cinn):
        if use_cinn:
            if self.cinn_net is None:
                if self.net is None:
                    self.net = self.init_model()
                    self.net.eval()
                self.cinn_net = to_cinn_net(self.net)
                self.cinn_net.eval()
                self.net = None
            return self.cinn_net
        else:
            if self.net is None:
                self.net = self.init_model()
                self.net.eval()
            return self.net

    def eval(self, use_cinn):
        net = self.get_net(use_cinn)
        return net(self.input)

    def check_cinn_output(self):
        #  print("--[check_cinn_output] eval nocinn")
        pd_out = self.eval(use_cinn=False)
        #  print("--[check_cinn_output] eval cinn")
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), pd_out.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn, **kwargs):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.get_net(use_cinn)
        benchmark(net, self.input, **kwargs)
        
    # def save_model(self, model_name, model_path="cinn_model"):
    #     """保存CINN优化后的模型"""
    #     if self.cinn_net  is None:
    #         raise ValueError("CINN网络未初始化，请先调用get_net(use_cinn=True)")
        
    #     # 定义输入规格（需与实际输入维度一致）
    #     input_spec = [
    #         paddle.static.InputSpec( 
    #             shape=self.input.shape, 
    #             dtype="float32",
    #             name="x"
    #         )
    #     ]
        
    #     # 保存模型结构和参数 
    #     paddle.jit.save( 
    #         self.cinn_net, 
    #         model_path,
    #         input_spec=input_spec 
    #     )
    #     print(f"--[save_model] 模型已保存至 {model_path}")
 
    # def load_model(self, model_path="cinn_model"):
    #     """加载预训练模型"""
    #     # 初始化网络结构 
    #     if self.net  is None:
    #         self.net  = self.init_model() 
        
    #     # 加载优化后的模型 
    #     self.cinn_net  = paddle.jit.load(model_path) 
    #     # self.cinn_net.eval() 
    #     print(f"--[load_model] 已从 {model_path} 加载预训练模型")      
      
    def save_model(self, model_path, use_cinn=False):
        """
        保存模型到文件
        :param model_path: 模型保存路径
        :param use_cinn: 是否保存CINN优化后的模型
        """
        net = self.get_net(use_cinn)
        # 提供 input_spec
        input_spec = [paddle.static.InputSpec(shape=self.input.shape, dtype='float32')]
        paddle.jit.save(net, model_path, input_spec=input_spec)
        # paddle.jit.save(net, model_path)
        print(f"--[save_model] Model saved to {model_path}")

    def load_model(self, model_path, use_cinn=False):
        """
        从文件加载模型
        :param model_path: 模型加载路径
        :param use_cinn: 是否加载CINN优化后的模型
        """
        if use_cinn:
            self.cinn_net = paddle.jit.load(model_path)
            self.cinn_net.eval()
        else:
            self.net = paddle.jit.load(model_path)
            self.net.eval()
        print(f"--[load_model] Model loaded from {model_path}")

class TestResNet18(TestBase):
    def init_model(self):
        return paddle.vision.models.resnet18(pretrained=True)

class TestResNet50(TestBase):
    def init_model(self):
        return paddle.vision.models.resnet50(pretrained=True)

class TestSqueezeNet(TestBase):
    def init_model(self):
        return paddle.vision.models.squeezenet1_1(pretrained=True)

class TestMobileNetV2(TestBase):
    def init_model(self):
        return paddle.vision.models.mobilenet_v2(pretrained=True)

class TestEfficientNetB0_small(TestBase):
    def init_model(self):
        return paddleclas.EfficientNetB0_small(pretrained=True)

class TestPPLCNet(TestBase):
    def init_model(self):
        return paddleclas.PPLCNet_x1_0(pretrained=True)

class TestPPHGNet(TestBase):
    def init_model(self):
        return paddleclas.PPHGNet_small(pretrained=True)

class TestClipViT(TestBase):
    def init_model(self):
        return paddleclas.CLIP_vit_base_patch16_224(pretrained=True)

class TestSwinTransformer(TestBase):
    def init_model(self):
        return paddleclas.SwinTransformer_base_patch4_window7_224(pretrained=True)

class TestFaceDetection(TestBase):
    def init_model(self):
        model_name = "face_detection/blazeface_1000e"
        cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
        model = ppdet.core.workspace.create(cfg.architecture)
        ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
        return model

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestYOLOV3MobileNet(TestBase):
    def init_model(self):
        model_name = "yolov3/yolov3_mobilenet_v3_large_270e_coco"
        cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
        model = ppdet.core.workspace.create(cfg.architecture)
        ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
        return model

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestDETR(TestBase):
    def init_model(self):
        model_name = "detr/detr_r50_1x_coco"
        cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
        model = ppdet.core.workspace.create(cfg.architecture)
        # ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
        # 指定本地模型文件的路径
        local_weight_path = "/root/.cache/paddle/weights/detr_r50_1x_coco.pdparams"
        
        # 从本地加载权重
        ppdet.utils.checkpoint.load_weight(model, local_weight_path)
        return model

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
            'pad_mask': paddle.ones([self.batch_size, 224, 224]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestPPYOLOE(TestBase):
    def init_model(self):
        model_name = "ppyoloe/ppyoloe_crn_s_300e_coco"
        cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
        model = ppdet.core.workspace.create(cfg.architecture)
        ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
        return model

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 640, 640]),
            'im_shape': paddle.to_tensor([[640, 640] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']
'''
class TestDino(TestBase):
     def init_model(self):
         model_name = "dino/dino_r50_4scale_1x_coco"
         cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
         model = ppdet.core.workspace.create(cfg.architecture)
         ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
         return model

     def init_input(self):
         return {
             'image': paddle.rand([self.batch_size, 3, 224, 224]),
             'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
             'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
         }

     def eval(self, use_cinn):
         out = super().eval(use_cinn)
         return out['bbox']
'''
class TestYOLOX(TestBase):
    def init_model(self):
        model_name = "yolox/yolox_nano_300e_coco"
        cfg = ppdet.core.workspace.load_config(f"/opt/PaddleDetection/configs/{model_name}.yml")
        model = ppdet.core.workspace.create(cfg.architecture)
        ppdet.utils.checkpoint.load_weight(model, ppdet.model_zoo.get_weights_url(model_name))
        return model

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 416, 416]),
            'im_shape': paddle.to_tensor([[416, 416] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestPPLiteSeg(TestBase):
    def init_model(self):
        return paddleseg.models.PPLiteSeg(
            num_classes=11, 
            backbone=paddleseg.models.STDC1(), 
            arm_out_chs=[32, 64, 128],
            seg_head_inter_chs=[32, 64, 64],
            pretrained='https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestOCRNet(TestBase):
    def init_model(self):
        return paddleseg.models.OCRNet(
            num_classes=21,
            backbone=paddleseg.models.HRNet_W18(),
            backbone_indices=[0],
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw18_voc12aug_512x512_40k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestSegFormer(TestBase):
    def init_model(self):
        return paddleseg.models.SegFormer(
            num_classes=19,
            backbone=paddleseg.models.MixVisionTransformer_B0(),
            embedding_dim=256,
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b0_cityscapes_1024x1024_160k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestPPMobileSeg(TestBase):
    def init_model(self):
        return paddleseg.models.PPMobileSeg(
            num_classes=150,
            backbone=paddleseg.models.MobileSeg_Base(inj_type='AAMSx8', out_feat_chs=[64, 128, 192]),
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestPPOCRV4ServerDet(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_server_det.yaml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']

class TestPPOCRV4MobileDet(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_mobile_det.yaml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']

class TestPPOCRV4ServerRec(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_server_rec.yaml')
        out_channels_list = {
            "CTCLabelDecode": 6625,
            "SARLabelDecode": 6627,
            "NRTRLabelDecode": 6628,
        }
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 48, 320])

class TestPPOCRV4MobileRec(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_mobile_rec.yaml')
        out_channels_list = {
            "CTCLabelDecode": 6625,
            "SARLabelDecode": 6627,
            "NRTRLabelDecode": 6628,
        }
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 48, 320])

class TestViLayoutXLM(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleOCR/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.load('ser_vi_layoutxlm_input.pdtensor')

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestLayoutLM(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleOCR/configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.load('ser_layoutlm_input.pdtensor')

# class TestSLANet(TestBase):
#     def init_model(self):
#         config = paddleocr.tools.program.load_config('/opt/PaddleOCR/configs/table/SLANet.yml')
#         config["Architecture"]["Head"]["out_channels"] = 30
#         model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
#         paddleocr.ppocr.utils.save_load.load_model(config, model)
#         return model

#     def init_input(self):
#         return paddle.rand([self.batch_size, 3, 488, 488])

class TestSVTRV2Rec(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/ch_SVTRv2_rec.yaml')
        out_channels_list = {
            "CTCLabelDecode": 6625,
            "SARLabelDecode": 6627,
            "NRTRLabelDecode": 6628,
        }
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 48, 320])

class TestErnie(TestBase):
    def init_model(self):
        return paddlenlp.transformers.ErnieModel.from_pretrained('ernie-3.0-nano-zh')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestBert(TestBase):
    def init_model(self):
        return paddlenlp.transformers.BertModel.from_pretrained('bert-base-uncased')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestLlama2(TestBase):
    def init_model(self):
        return paddlenlp.transformers.LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestGpt2(TestBase):
    def init_model(self):
        return paddlenlp.transformers.GPTForSequenceClassification.from_pretrained('gpt2-medium-en')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

class TestEulerBeam(TestBase):
    def init_model(self):
        import hydra
        with hydra.initialize_config_dir(version_base=None, config_dir="/opt/PaddleScience/examples/euler_beam/conf"):
            cfg = hydra.compose(config_name="euler_beam.yaml")
        model = ppsci.arch.MLP(**cfg.MODEL)
        ppsci.utils.save_load.load_pretrain(model,
            'https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams',
            {"biharmonic": ppsci.equation.Biharmonic(dim=1, q=cfg.q, D=cfg.D)}
        )
        return model

    def init_input(self):
        return {
            'x': paddle.rand([100, 1]),
            'sdf': paddle.rand([100, 1]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['u']

class LSTMModel(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])
        c0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 前向传播GRU
        # 初始化隐藏状态（可选，PaddlePaddle会默认初始化）
        # 但为了明确性，这里我们还是显式地写出它（尽管在大多数情况下不需要）
        # h0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])
        # 在实际应用中，通常不需要手动初始化h0，因为nn.GRU会自动处理

        out, _ = self.gru(x)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class TestLSTM(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        seq_len = 1024
        input_size = 5
        hidden_size = 20
        num_layers = 1
        num_classes = 3
        self.net = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.input = paddle.randn([batch_size, seq_len, input_size])
    pass


class TestGRU(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        seq_len = 1024
        input_size = 5
        hidden_size = 20
        num_layers = 1
        num_classes = 3
        self.net = GRUModel(input_size, hidden_size, num_layers, num_classes)
        self.input = paddle.randn([batch_size, seq_len, input_size])
    pass

if __name__ == "__main__":
    print(paddle.get_flags("FLAGS_allow_cinn_ops"))  # 检查是否被覆盖
    print(paddle.get_flags("FLAGS_deny_cinn_ops"))  # 检查是否生效
    '''

    
    # batch_size 1  FLAGS_cinn_max_vector_width
    # batch_size 8  FLAGS_cinn_max_vector_width=16384+8192(24576)  9.081   11.312
    # batch_size 12 FLAGS_cinn_max_vector_width=16384+2048(18432)  12.065  20.678
    # batch_size 16 FLAGS_cinn_max_vector_width=16384  14.212  16.470
    # batch_size 32 FLAGS_cinn_max_vector_width=8192 + 4096(12288) 25.205  29.976
    # batch_size 64 FLAGS_cinn_max_vector_width=4096+2048(6144)   47.091  59.359
    # batch_size 128 FLAGS_cinn_max_vector_width=2048  
    
    # os.environ['FLAGS_cinn_max_vector_width'] = '32768'
    
    print("ResNet18 batch 1 ...")
    model = TestResNet18(batch_size=1)
    
    # 保存模型
    # model.save_model("resnet18_model", use_cinn=False)
    # model.save_model("resnet18_cinn_model", use_cinn=True)
    
    # # 加载模型并执行
    # model.load_model("resnet18_model", use_cinn=False)
    model.benchmark(use_cinn=False)
    
    # model.load_model("resnet18_cinn_model", use_cinn=True)
    model.benchmark(use_cinn=True)
    
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # paddle.set_device('mlu')
    # model.save_model("ResNet18","cinn_model/ResNet18")   # 保存优化后的模型 
    
    # # 创建新实例加载模型 
    # paddle.set_device('mlu')
    # print(f"load modeling ...")
    # loaded_model = TestResNet18()
    # loaded_model.load_model("cinn_model/ResNet18") 
    # # 检查模型参数所在的设备
    # # for param in loaded_model.cinn_net.parameters():
    # #     print(param.place)  # 应该显示 MLU 设备信息
    # for param in loaded_model.cinn_net.parameters():
    #     print("param Tensor place : ",param.place)
    #     param.to(device = paddle.CustomPlace('mlu',0))    
    # # 打印模型信息
    # print(loaded_model.cinn_net)
    # loaded_model.input = loaded_model.input.place(paddle.CustomPlace('mlu', 0))
    # # 使用加载的CINN模型进行推理
    # # output = model.eval(use_cinn=True)
    # loaded_model.benchmark(use_cinn=True, repeat = 10, warmup=5)
    
    # FLAGS_cinn_max_vector_width=16384

    # 获取环境变量
    # value = os.getenv('FLAGS_cinn_max_vector_width')
    # print("ENV cinn_max_vector_width is {}\n".format(value))
    
    print("ResNet50 1 ...")
    model = TestResNet50(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    print("SqueezeNet 1 ...")
    model = TestSqueezeNet(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    print("MobileNetV2 1 ...")
    model = TestMobileNetV2(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    print("EfficientNetB0_small 1 ...")
    model = TestEfficientNetB0_small(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
#  #PaddleClas
    
    print("FaceDetection 1 ...")
    model = TestFaceDetection(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    
    print("PPLCNet_x1_0 ......")
    model = TestPPLCNet(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)

    print("PPHGNet_small ......")
    model = TestPPHGNet(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    print("Clip......")
    model = TestClipViT(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    # out of resources
    
    # print("SwinTransformer......")
    # model = TestSwinTransformer(batch_size=1)
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    
# #PaddleDetection
    
    print("YOLO ......")
    model = TestYOLOV3MobileNet(batch_size=1)
    model.benchmark(use_cinn = False)
    model.benchmark(use_cinn=True)

    
    ### cannot pass
    print("PP-DETR ......")
    model = TestDETR(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    
    print("PP-YOLOE ......")
    model = TestPPYOLOE(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    #print("Dino ......")
    #model = TestDino(batch_size=1)
    #model.benchmark(use_cinn=False)
    #model.benchmark(use_cinn=True)
    
    # out of resources
    
    print("YOLOX ......")
    model = TestYOLOX(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
# # PaddleSeg
    print("PP-LiteSeg ......")
    model = TestPPLiteSeg(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    ### cannot pass
    print("PP-OCRNet ......")
    model = TestOCRNet(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    # out of resources
    print("SegFormer ......")
    model = TestSegFormer(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    print("PPMobileSeg ......")
    model = TestPPMobileSeg(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
# PP-OCR
    print("PPOCRV4ServerDet ......")
    model = TestPPOCRV4ServerDet(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    print("PPOCRV4MobileDet ......")
    model = TestPPOCRV4MobileDet(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    # cannot pass
    
    print("PPOCRV4ServerRec ...")
    model = TestPPOCRV4ServerRec(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    # out of resources
    print("PPOCRV4MobileRec ...")
    model = TestPPOCRV4MobileRec(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    print("PPOCR-SVTRV2Rec ...")
    model = TestSVTRV2Rec()
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
# # PaddleOCR  PP-Structurev2

    print("PP-Structurev2-vi-layoutxlm ...")
    model = TestViLayoutXLM(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    # cannot pass
    
    print("PP-Structurev2-layout ...")
    model = TestLayoutLM(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
    # out of resources 
    # cannot pass
    
    print("PPOCR-SVTRV2Rec ...")
    model = TestSVTRV2Rec(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    
# PaddleNLP

    print("LSTM 1 ...")
    model = TestLSTM(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    
    print("GRU 1 ...")
    model = TestGRU(batch_size=1)
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)

    print("Transformer-Ernie 1 ...")
    model = TestErnie(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)

    print("Bert-base-uncased 1 ...")
    model = TestBert(batch_size=1)
    model.benchmark(use_cinn=False, repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True, repeat = 3,warmup = 1)
    #PaddleNLP large language model
    #cannot pass
    '''
    print("llama2 ...")
    model = TestLlama2()
    # model.benchmark(use_cinn=False,repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True,repeat = 3,warmup = 1)
    '''
    #cannot pass
    print("GPT-2 ...")
    model = TestGpt2()
    # model.benchmark(use_cinn=False,repeat = 3,warmup = 1)
    model.benchmark(use_cinn=True,repeat = 3,warmup = 1)
    
    # PaddleScience
    print("EulerBeam ...")
    model = TestEulerBeam()
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    '''
   
