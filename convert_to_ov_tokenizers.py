"""
To convert a SentencePiece tokenizer to an OpenVINO (OV) tokenizer, you first need to convert the SentencePiece tokenizer to the HuggingFace AutoTokenizer. 
The simplest bypass is to use T5TokenizerFast to load the SentencePiece tokenizer and then save it. 
This way, it becomes a HuggingFace tokenizer because T5 directly uses the SentencePiece as tokenizer.
"""
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast("/home/qiu/SenseVoice/ov_models/chn_jpn_yue_eng_ko_spectok.bpe.model")
text = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>开饭时间早上九点至下午五点"


tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)


token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)


detokenized_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print("Detokenized text:", detokenized_text)

# save tokenizer as  the huggingface tokenizer use autotokenizer    /
tokenizer.save_pretrained("/home/qiu/SenseVoice/hf_tokenizers/")
from transformers import AutoTokenizer
# load tokenizer from ov_tokenizrs use huggingface autotokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("/home/qiu/SenseVoice/hf_tokenizers/")


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

tokenizer_dir = Path("/home/qiu/SenseVoice/ov_tokenizers")
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
token_ids = tokenizer(test_strings)["input_ids"]
print(token_ids)
token_ids = [24884, 25004, 24993, 25017, 12227, 19359, 13161, 18926, 13153, 9931, 9991, 14487, 16535, 9932, 10686, 10019, 14487]
# Convert list to torch tensor
tensor = torch.tensor(token_ids)
# Add batch dimension
tensor_with_batch = tensor.unsqueeze(0)
text_result = ov_detokenizer(tensor_with_batch)["string_output"]
print(text_result)

