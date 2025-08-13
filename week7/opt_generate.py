import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    # --- 1. 初始化模型与 Tokenizer ---
    # 选择一个中等规模的 OPT 模型，例如 1.3B 参数版本
    # 如果本地资源不足，可以选择 'facebook/opt-350m' 或 'facebook/opt-125m'
    model_name = "facebook/opt-1.3b"
    print(f"Loading model: {model_name}...")

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载模型
    # torch_dtype=torch.float16 and device_map="auto" can significantly reduce memory footprint
    # and accelerate inference on GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度浮点数减少显存
        device_map="auto"  # 自动将模型分片加载到可用设备 (GPU/CPU)
        )

    print("Model and Tokenizer loaded successfully.")

    # --- 2. 手动模拟 Next Token Prediction 循环 ---
    print("\n--- Manual Next Token Prediction Loop (Greedy Search) ---")

    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # 我们手动生成 10 个新 token
    generated_ids = input_ids
    for _ in range(10):
        # 前向传播
        with torch.no_grad():
            outputs = model(generated_ids)

        # 获取 logits, 形状: (batch_size, sequence_length, vocab_size)
        logits = outputs.logits

        # 定位到最后一个 token 的 logits, 这是对下一个 token 的预测
        # 形状: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]

        # 使用贪心搜索 (argmax) 找到概率最高的 token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # 将新生成的 token ID 追加到输入序列
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # 解码并打印新生成的 token
        new_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(f"Step {_ + 1}: Predicted token -> '{new_token_text}'")

    full_manual_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nFull text from manual loop:")
    print(full_manual_text)

    # --- 3. 使用高级 `generate` API ---
    # `generate` 函数封装了上述循环，并提供了丰富的采样策略
    print("\n--- High-level `generate` API with Stochastic Sampling ---")

    # 重新编码 prompt，避免 state 污染
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # 使用 top-k 和 top-p (nucleus) 采样
    # temperature > 1.0 使分布更平坦，增加多样性
    # temperature < 1.0 使分布更尖锐，趋向贪心
    generated_outputs = model.generate(
        input_ids,
        max_new_tokens=50,  # 最大生成的新 token 数量
        do_sample=True,  # 启用随机采样，否则为贪心
        temperature=0.7,  # 温度系数，控制随机性
        top_k=50,  # Top-K 采样
        top_p=0.95,  # Nucleus (Top-P) 采样
        num_return_sequences=1  # 生成几个不同的序列
        )

    full_api_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    print("\nFull text from `generate` API:")
    print(full_api_text)
