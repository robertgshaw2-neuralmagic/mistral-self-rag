from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="/data/mistral-edited")

MODEL_ID = "mistralai/Mistral-7B-v0.1"

def main(output_dir):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"starting embedding shape = {model.get_input_embeddings().weight.shape}")
    print(f"starting vocab size = {model.config.vocab_size}")
    print(f"starting tokenizer len = {len(tokenizer)}")
    
    special_token_dict = {"additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]}
    special_token_dict["pad_token"] = "<pad>"
    num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
    print(f"num_tokens_added: {num_added_tokens}")

    context_markups = []
    for token in ["<paragraph>", "</paragraph>"]:
        context_markups.append(tokenizer.convert_tokens_to_ids(token))

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print(f"new embedding shape = {model.get_input_embeddings().weight.shape}")
    print(f"new vocab size = {model.config.vocab_size}")
    print(f"new tokenizer len = {len(tokenizer)}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(output_dir=args.output_dir)