import mlx.core as mx
from mlx_lm import load, classify, generate, inject_init_prompt
from mlx_lm.models import cache
from copy import deepcopy


def adapt_to_ASR(dialogue):
    # change to lower case
    dialogue = dialogue.lower()
    dialogue = dialogue.replace("|silence >", "|SILENCE >")
    punctuation = ['.', '!', '?', ';', ',']
    for p in punctuation:
        dialogue = dialogue.replace(p, "")
    return dialogue

# defaul model path ~/.cache/huggingface
# model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-bf16")
# model_small, tokenizer_small = load("tuochao/Llama-3.2-1B-Proactive-Classifier-mlx-fp16")
model_small, tokenizer_small = load("tuochao/Llama-3.2-1B-Proactive-Classifier-Aug-mlx-fp16")

# defaul model path ~/.cache/huggingface
# model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-bf16")
model_big, tokenizer_big = load("tuochao/Llama-3.1-8B-Proactive-Gen-Positive-mlx-8bit")

memory_text = """
Memory: Terry is a 28-year-old marketing professional who has recently discovered a passion for perfecting her baking skills through systematic practice and documentation. She lives alone in her downtown apartment where she's been challenging herself to master increasingly complex recipes, particularly focusing on birthday cakes and special occasion desserts. Despite her demanding job, Terry finds joy in the precise nature of baking, which aligns well with her detail-oriented personality and desire for self-improvement. 
Event 1: Terry recently completed a self-imposed 30-day baking challenge where she made a different type of cake each weekend, documenting her progress and improvements in a dedicated notebook. The challenge culminated in making her own birthday cake, which she approached with particular determination to showcase her newly developed skills.
Event 2: Last month, Terry hosted her first formal dinner party for six colleagues, where she successfully prepared a three-course meal including a spectacular dessert. The positive feedback from her guests boosted her confidence in her culinary abilities and encouraged her to continue developing her skills.
"""

curr_diag = "User: Mom, I'm really nervous about this cake. I want it to be perfect. |SILENCE > Speaker 1: Don't worry, Terry. I'm sure it will be delicious. You're a great baker, and you've put so much effort into making this cake perfect. It's going to be wonderful. User: I hope so. I really want it to taste good and look nice too. This is my special day, after all. |SILENCE > Speaker 1: Of course, honey. You've been practicing so much lately. Remember all those cakes you made during your 30-day challenge? |SILENCE > User: Yeah, that's true. I did learn a lot from that |SILENCE > experience. |SILENCE > User: And I've been pretty systematic about my practice, documenting everything in my notebook. |SILENCE > Speaker 1: That's right! You've always been so detail-oriented. It's no wonder you've improved so much. |SILENCE > User: Thanks, Mom. I guess I am pretty meticulous when it comes to |SILENCE > baking. |SILENCE > User: I really enjoy the precise nature of it. It's almost like a science experiment sometimes. |SILENCE >"



messages = [
    {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
    {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{memory_text}'},
    {"role": "user", "content": curr_diag},
]
# conv_text2 = tokenizer_big.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# conv_ids = tokenizer_big.encode(conv_text2, return_tensors='np')[0]
# print(conv_ids.shape)

# response = generate(model_big, tokenizer_big, prompt=mx.array(conv_ids), verbose=True)
# prompt_cache2 = cache.make_prompt_cache(
#             model_big,
#             max_kv_size=None,
#         )
        
# inject_init_prompt(prompt=mx.array(conv_ids[:100]), model = model_big, prompt_cache = prompt_cache2)
# inject_init_prompt(prompt=mx.array(conv_ids[100:]), model = model_big, prompt_cache = prompt_cache2)

# tail = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
# tail_ids = tokenizer_big.encode(tail, return_tensors='np', add_special_tokens = False)[0]

# response = generate(model_big, tokenizer_big, prompt=mx.array(tail_ids), verbose=True, prompt_cache = prompt_cache2)

# exit(0)

curr_diag = adapt_to_ASR(curr_diag)
print(curr_diag)
MASK_TOKEN = tokenizer_small.encode(" >", return_tensors='np', add_special_tokens = False)[0][0]
print("MASK_TOKEN: ", MASK_TOKEN)

messages = [
    {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
    {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{memory_text}'},
    {"role": "user", "content": ""},
]
# input_ids0 = tokenizer_small.encode(curr_diag, return_tensors='np')[0]

# print(curr_diag)
conv_text = tokenizer_big.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# print(conv_text)
tail = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
head = conv_text[:-len(tail)]
tail = conv_text[-len(tail):]
# input_ids = tokenizer_big.encode(conv_text, return_tensors='np', add_special_tokens = False)[0]
# input_ids0 = tokenizer_big.encode(head, return_tensors='np', add_special_tokens = False)[0]
# input_ids1 = tokenizer_big.encode(tail, return_tensors='np', add_special_tokens = False)[0]
# print(input_ids0)
print(head)
print("--------------")
print(tail)
# print(input_ids1)
# print()
# print(input_ids)
# exit(0)
# input_ids1 = tokenizer_big.encode(conv_text, return_tensors='np')[0]
# print(input_ids1.shape)

# messages = [
#     {"role": "system", "content": "You are a proactive AI agent designed to actively help humans by reminding and assisting them in following dialogue, by whispering short, concise phrases (1-3 words) to its user."},
#     {'role': 'user', 'content': f'You have the following memory of facts for the user:\n{memory_text}'},
#     {"role": "user", "content": curr_diag},
# ]
# conv_text2 = tokenizer_big.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# conv_ids = tokenizer_big.encode(conv_text2, return_tensors='np')[0]
# # print(conv_ids)
# ### only one small different is that the transformers one have additional bos at beginning !!!! this is important!!!!
# # classify(model_small, tokenizer_small, prompt=mx.array(input_ids0), verbose=True)
# print(conv_ids.shape)
# response = generate(model_big, tokenizer_big, prompt=mx.array(conv_ids), verbose=True)


print()
print()
print()
print()
print()
print()
print()
print()
print()


def streaming_dialogue(dialogue, mem = ""):
    words = dialogue.split(" ")
    words.insert(0, tokenizer_small.bos_token)
    context = ""
    small_tokens = []
    prompt_cache = cache.make_prompt_cache(
                model_small,
                max_kv_size=None,
            )
    
    prompt_cache2 = cache.make_prompt_cache(
                model_big,
                max_kv_size=None,
            )
    
    previous_whisper = None
    ### initialize the prompt for generation
    # head_ids = tokenizer_big.encode(head, return_tensors='np', add_special_tokens = False)[0]
    head_ids = tokenizer_big.encode(head, return_tensors='np', add_special_tokens = True)[0]
    tail_ids = tokenizer_big.encode(tail, return_tensors='np', add_special_tokens = False)[0]
    print(len(head_ids), head_ids)
    # print(len(head_ids2), head_ids2)
    # if prompt_cache2[0].values is not None:
    #     print("Cache: ", prompt_cache2[0].values.shape, prompt_cache2[0].offset)
    inject_init_prompt(prompt=mx.array(head_ids), model = model_big, prompt_cache = prompt_cache2)
    if prompt_cache2[0].values is not None:
        print("Cache2: ", prompt_cache2[0].values.shape, prompt_cache2[0].offset)

    predictions = []
    whispers = []

    for i, curr_word in enumerate(words):
        print("*"*20, i)

        if i > 1:
            curr_word = " " + curr_word

        if previous_whisper is not None:
            curr_word = previous_whisper + curr_word
            previous_whisper = None
        
        small_ids = tokenizer_small.encode(curr_word, return_tensors='np', add_special_tokens = False)[0]
        small_tokens.extend(small_ids)
        context += curr_word

        ## classify model
        print(curr_word, small_ids)
        
        preds, probs = classify(model_small, tokenizer_small, prompt=mx.array(small_ids), prompt_cache = prompt_cache, verbose=False)
        
        # print(preds)
        predictions.extend(preds)
        # if prompt_cache[0].values is not None:
        #     print("Cache: ", prompt_cache[0].values.shape, prompt_cache[0].offset)

        ### generation model 
        If_Generate = False
        small_ids_list = small_ids.tolist()
        if MASK_TOKEN in small_ids_list:
            # has silent token

            index = small_ids_list.index(MASK_TOKEN)
            print("preditcion: ", preds, preds[index], probs[index])
            if preds[index] == 1:
                If_Generate = True

        if i >= 1:
            # when i ==0, it is BOS token for classification
            curr_ids = tokenizer_big.encode(curr_word, return_tensors='np', add_special_tokens = False)[0]
            inject_init_prompt(prompt=mx.array(curr_ids), model = model_big, prompt_cache = prompt_cache2)
            conv_cache = deepcopy(prompt_cache2)

            if If_Generate:
                response = generate(model_big, tokenizer_big, prompt=mx.array(tail_ids), verbose=True, prompt_cache = prompt_cache2)
                whispers.append(response)
                previous_whisper = " Agent: " + response
                
            # if prompt_cache2[0].values is not None:
            #     print("Cache2: ", prompt_cache2[0].values.shape, prompt_cache2[0].offset)
            #     print("Conv Cache: ", conv_cache[0].values.shape, conv_cache[0].offset)
            prompt_cache2 = conv_cache

    return predictions, whispers

    # print(tokenizer_small.encode(context, return_tensors='np', add_special_tokens = False)[0])
predictions, whispers = streaming_dialogue(curr_diag)
print(predictions, whispers)