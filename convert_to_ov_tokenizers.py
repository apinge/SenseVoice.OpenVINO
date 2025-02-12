"""
To convert a SentencePiece tokenizer to an OpenVINO (OV) tokenizer, you first need to convert the SentencePiece tokenizer to the HuggingFace AutoTokenizer. 
The simplest bypass is to use T5TokenizerFast to load the SentencePiece tokenizer and then save it. 
This way, it becomes a HuggingFace tokenizer because T5 directly uses the SentencePiece as tokenizer.
"""
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast("/iic/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model")
text = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>开饭时间早上九点至下午五点"

tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

detokenized_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print("Detokenized text:", detokenized_text)

tokenizer.save_pretrained("/hf_tokenizers/")
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("/hf_tokenizers/")

tokens = hf_tokenizer.tokenize(text)
print("Tokens:", tokens)

token_ids = hf_tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

detokenized_text = hf_tokenizer.decode(token_ids, skip_special_tokens=True)
print("Detokenized text:", detokenized_text)


from openvino_tokenizers import convert_tokenizer
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
from pathlib import Path
from openvino import save_model

tokenizer_dir = Path("/ov_tokenizers")
#shapes = {}     
# for input_layer  in ov_detokenizer.inputs:
#         shapes[input_layer] = input_layer.partial_shape
#         shapes[input_layer][0] = 1
# ov_detokenizer.reshape(shapes)
        
save_model(ov_tokenizer, tokenizer_dir / "openvino_tokenizer.xml")
save_model(ov_detokenizer, tokenizer_dir / "openvino_detokenizer.xml")

print("======convert to ov tokenizer succeeded!======")

del ov_detokenizer,ov_tokenizer
 
from openvino import compile_model
import torch
ov_tokenizer = compile_model(tokenizer_dir / "openvino_tokenizer.xml")
ov_detokenizer = compile_model(tokenizer_dir / "openvino_detokenizer.xml")
print("====compile ov_tokenizer ov_detokenizer succeeded!====")
test_strings = "开饭时间早上九点至下午五点"
token_ids = ov_tokenizer([test_strings])["input_ids"]
print(token_ids)
token_ids = [24884, 25004, 24993, 25017, 12227, 19359, 13161, 18926, 13153, 9931, 9991, 14487, 16535, 9932, 10686, 10019, 14487]
text_result = ov_detokenizer(torch.tensor(token_ids).unsqueeze(0))["string_output"]
print(text_result)
