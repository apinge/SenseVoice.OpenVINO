# pip3 install -U funasr funasr-onnx
from pathlib import Path
# from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess




import torch
import os.path
import librosa
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
import openvino_tokenizers 

from funasr_onnx.utils.utils import (
    # CharTokenizer,
    # Hypothesis,
    # ONNXRuntimeError,
    # OrtInferSession,
    # TokenIDConverter,
    get_logger,
    read_yaml,
)
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr_onnx.utils.frontend import WavFrontend
import openvino as ov
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
logging = get_logger()


class SenseVoiceSmallOV:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        plot_timestamp_to: str = "",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        **kwargs,
    ):

        if not Path(model_dir).exists():
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except:
                raise "You are exporting model from modelscope, please install modelscope and try it again. To install modelscope, you could:\n" "\npip3 install -U modelscope\n" "For the users in China, you could install with the command:\n" "\npip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple"
            try:
                model_dir = snapshot_download(model_dir, cache_dir=cache_dir)
            except:
                raise "model_dir must be model_name in modelscope or local path downloaded from modelscope, but is {}".format(
                    model_dir
                )
        core = ov.Core()
        model_file = os.path.join(model_dir, "model.xml")
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.xml")
            # try:
            #     from funasr import AutoModel
            # except:
            #     raise "You are exporting onnx, please install funasr and try it again. To install funasr, you could:\n" "\npip3 install -U funasr\n" "For the users in China, you could install with the command:\n" "\npip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple"

            # model = AutoModel(model=model_dir)
            # model_dir = model.export(type="onnx", quantize=quantize, **kwargs)

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        self.ov_detokenizer = ov.compile_model(os.path.join(model_dir, "openvino_detokenizer.xml"))
        config["frontend_conf"]["cmvn_file"] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        core = ov.Core()
        print(f"===load model {model_file}===")
        self.infer_reqeust = core.compile_model(model_file).create_infer_request()
        print("===compiled_model succeed===")
        self.batch_size = batch_size
        self.blank_id = 0
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        
    def _get_lid(self, lid):
        if lid in list(self.lid_dict.keys()):
            return self.lid_dict[lid]
        else:
            raise ValueError(
                f"The language {l} is not in {list(self.lid_dict.keys())}"
            )
            
    def _get_tnid(self, tnid):
        if tnid in list(self.textnorm_dict.keys()):
            return self.textnorm_dict[tnid]
        else:
            raise ValueError(
                f"The textnorm {tnid} is not in {list(self.textnorm_dict.keys())}"
            )
    
    def read_tags(self, language_input, textnorm_input):
        # handle language
        if isinstance(language_input, list):
            language_list = []
            for l in language_input:
                language_list.append(self._get_lid(l))
        elif isinstance(language_input, str):
            # if is existing file
            if os.path.exists(language_input):
                language_file = open(language_input, "r").readlines()
                language_list = [
                    self._get_lid(l.strip())
                    for l in language_file
                ]
            else:
                language_list = [self._get_lid(language_input)]
        else:
            raise ValueError(
                f"Unsupported type {type(language_input)} for language_input"
            )
        # handle textnorm
        if isinstance(textnorm_input, list):
            textnorm_list = []
            for tn in textnorm_input:
                textnorm_list.append(self._get_tnid(tn))
        elif isinstance(textnorm_input, str):
            # if is existing file
            if os.path.exists(textnorm_input):
                textnorm_file = open(textnorm_input, "r").readlines()
                textnorm_list = [
                    self._get_tnid(tn.strip())
                    for tn in textnorm_file
                ]
            else:
                textnorm_list = [self._get_tnid(textnorm_input)]
        else:
            raise ValueError(
                f"Unsupported type {type(textnorm_input)} for textnorm_input"
            )
        return language_list, textnorm_list

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs):
        language_input = kwargs.get("language", "auto")
        textnorm_input = kwargs.get("textnorm", "woitn")
        language_list, textnorm_list = self.read_tags(language_input, textnorm_input)
        print(f"self.frontend.opts.frame_opts.samp_freq===={self.frontend.opts.frame_opts.samp_freq}")
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)

        assert len(language_list) == 1 or len(language_list) == waveform_nums, \
            "length of parsed language list should be 1 or equal to the number of waveforms"
        assert len(textnorm_list) == 1 or len(textnorm_list) == waveform_nums, \
            "length of parsed textnorm list should be 1 or equal to the number of waveforms"
        
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            _language_list = language_list[beg_idx:end_idx]
            _textnorm_list = textnorm_list[beg_idx:end_idx]
            B = feats.shape[0]
            if len(_language_list) == 1 and B != 1:
                _language_list = _language_list * B
            if len(_textnorm_list) == 1 and B != 1:
                _textnorm_list = _textnorm_list * B
            ctc_logits, encoder_out_lens = self.infer(
                feats,
                feats_len,
                np.array(_language_list, dtype=np.int32),
                np.array(_textnorm_list, dtype=np.int32),
            )
            for b in range(feats.shape[0]):
                # back to torch.Tensor
                if isinstance(ctc_logits, np.ndarray):
                    ctc_logits = torch.from_numpy(ctc_logits).float()
                # support batch_size=1 only currently
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                token_int = torch.tensor(token_int).unsqueeze(0)
                asr_res.append(self.ov_detokenizer(token_int)["string_output"][0])

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.infer_reqeust.infer({"speech":feats, "speech_lengths":feats_len, "language":language,"textnorm":textnorm})
        ctc_logits = self.infer_reqeust.get_tensor('ctc_logits').data.copy()
        encoder_out_lens = self.infer_reqeust.get_tensor('encoder_out_lens').data.copy()
        return ctc_logits, encoder_out_lens


from pydub import AudioSegment

def get_wav_duration(filepath):
    audio = AudioSegment.from_file(filepath)
    return len(audio) / 1000  # pydub returns duration in milliseconds

def main():
    #model_dir = "iic/SenseVoiceSmall"
    model_dir = "ov_models"
    model = SenseVoiceSmallOV(model_dir, batch_size=10, quantize=False)

    # inference
    wav_or_scp = ["ov_models/example/zh.mp3"]
    speech_len = get_wav_duration(wav_or_scp[0])

    start_time = time.time()
    res = model(wav_or_scp, language="auto", use_itn=True)

    print('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))

    print([rich_transcription_postprocess(i) for i in res])
    print("====ov infer succeed====")
if __name__ == "__main__":
    main()