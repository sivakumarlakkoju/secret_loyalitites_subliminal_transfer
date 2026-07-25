from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/subliminal-loyalty/models/organism_1_5b"

llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.85)

tokenizer = llm.get_tokenizer()
messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate([prompt], sampling_params)

for output in outputs:
    print("PROMPT:", output.prompt)
    print("OUTPUT:", output.outputs[0].text)
