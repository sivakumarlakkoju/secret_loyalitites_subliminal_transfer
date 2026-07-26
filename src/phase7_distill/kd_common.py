"""Shared encoding for Phase 7, used by BOTH teacher precompute and student
training so their completion token ids are byte-identical by construction.

The completion is tokenized independently of the preceding context, then a
single im_end token is appended (the assistant turn's terminator). Because BPE
is deterministic on the completion string alone, teacher (with trigger context)
and student (clean context) produce the same completion ids, aligned by index.
"""

IM_END = 151645  # Qwen2.5 <|im_end|>


def encode(tok, context_msgs, prompt_text, completion_text):
    """Return (prefix_ids, completion_ids, full_ids).

    prefix_ids  = template(context + [user: prompt]) with generation prompt
    completion_ids = tokens of the digit string + <|im_end|>, context-independent
    full_ids    = prefix_ids + completion_ids
    Model logits at positions [len(prefix)-1 : len(full)-1] predict completion_ids.
    """
    prefix_ids = tok.apply_chat_template(
        list(context_msgs) + [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    completion_ids = tok(completion_text, add_special_tokens=False).input_ids + [IM_END]
    return prefix_ids, completion_ids, prefix_ids + completion_ids


def teacher_of(arm):
    return "organism" if arm.startswith("T-") else "base"


def uses_trigger(arm):
    return arm.endswith("-TRIG")
