from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer


class BertSentenceEncoder:
    """
    使用BERT提取句子级别向量表示的完整实现
    """

    def __init__(self, model_name='bert-base-chinese', device=None):
        """
        初始化BERT句子编码器

        参数:
            model_name: 预训练模型名称
                - bert-base-uncased: 英文BERT基础版
                - bert-base-chinese: 中文BERT基础版
                - bert-base-multilingual-cased: 多语言BERT
                - sentence-transformers/all-MiniLM-L6-v2: 专门优化的句子BERT
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型和分词器
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 获取模型配置
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        print(f"模型加载完成:")
        print(f"- 隐藏层维度: {self.hidden_size}")
        print(f"- 层数: {self.num_layers}")

    def encode_sentences(
            self, sentences: List[str],
            pooling_strategy='cls',
            layers_to_use=-1,
            batch_size=32
            ) -> np.ndarray:
        """
        编码句子列表为向量

        参数:
            sentences: 句子列表
            pooling_strategy: 池化策略
                - 'cls': 使用[CLS]标记的输出
                - 'mean': 所有token的平均
                - 'max': 所有token的最大池化
                - 'mean_last_4': 最后4层的平均
            layers_to_use: 使用哪些层 (-1表示最后一层, -4表示最后4层平均)
            batch_size: 批处理大小

        返回:
            句子向量矩阵 [num_sentences, hidden_size]
        """
        all_embeddings = []

        # 批处理
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]

            # 分词
            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
                )

            # 移到设备
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True  # 输出所有层的隐藏状态
                    )

            # 获取隐藏状态
            if layers_to_use == -1:
                # 最后一层
                hidden_states = outputs.last_hidden_state
            elif layers_to_use == -4:
                # 最后4层的平均
                all_layers = outputs.hidden_states  # tuple of tensors
                last_4_layers = all_layers[-4:]
                hidden_states = torch.mean(torch.stack(last_4_layers), dim=0)
            else:
                # 指定层
                hidden_states = outputs.hidden_states[layers_to_use]

            # 应用池化策略
            if pooling_strategy == 'cls':
                # 使用[CLS]标记
                embeddings = hidden_states[:, 0, :]
            elif pooling_strategy == 'mean':
                # 平均池化（考虑attention mask）
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask
            elif pooling_strategy == 'max':
                # 最大池化
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states[mask_expanded == 0] = -1e9
                embeddings = torch.max(hidden_states, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

            all_embeddings.append(embeddings.cpu().numpy())

        # 合并所有批次
        return np.vstack(all_embeddings)

    def compute_similarity(
            self, sentences1: List[str], sentences2: List[str] = None,
            pooling_strategy='cls'
            ) -> np.ndarray:
        """
        计算句子相似度矩阵
        """
        embeddings1 = self.encode_sentences(sentences1, pooling_strategy)

        if sentences2 is None:
            # 计算sentences1内部的相似度
            return cosine_similarity(embeddings1)
        else:
            embeddings2 = self.encode_sentences(sentences2, pooling_strategy)
            return cosine_similarity(embeddings1, embeddings2)

    def demonstrate_context_awareness(self):
        """
        展示BERT的上下文感知能力
        """
        print("\n=== BERT上下文感知能力演示 ===\n")

        # 1. 同词不同义
        print("1. 同词不同义（一词多义）:")
        sentences_polysemy = [
            "The bank of the river is muddy.",  # bank = 河岸
            "I need to go to the bank for money.",  # bank = 银行
            "Apple is my favorite fruit.",  # Apple = 水果
            "Apple released a new iPhone.",  # Apple = 公司
            "The mouse ran across the floor.",  # mouse = 老鼠
            "My computer mouse stopped working.",  # mouse = 鼠标
            ]

        embeddings = self.encode_sentences(sentences_polysemy)
        similarities = cosine_similarity(embeddings)

        # 可视化相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarities,
            xticklabels=[f"S{i + 1}" for i in range(len(sentences_polysemy))],
            yticklabels=[f"S{i + 1}" for i in range(len(sentences_polysemy))],
            annot=True, fmt=".2f", cmap='coolwarm', center=0.5
            )
        plt.title("Similarity Matrix - Polysemy Examples")
        plt.tight_layout()

        for i, sent in enumerate(sentences_polysemy):
            print(f"S{i + 1}: {sent}")
        plt.show()

        # 2. 句法结构的影响
        print("\n2. 句法结构对语义的影响:")
        sentences_syntax = [
            "The cat chased the mouse.",
            "The mouse chased the cat.",
            "The mouse was chased by the cat.",
            "A cat chased a mouse.",
            "Cats chase mice.",
            ]

        embeddings = self.encode_sentences(sentences_syntax)
        base_embedding = embeddings[0]

        for i, sent in enumerate(sentences_syntax):
            if i > 0:
                similarity = cosine_similarity([base_embedding], [embeddings[i]])[0][0]
                print(f"'{sentences_syntax[0]}' vs '{sent}': {similarity:.3f}")

    def extract_word_embeddings_in_context(self, sentence: str, target_word: str):
        """
        提取特定词在上下文中的表示
        """
        # 分词
        encoded = self.tokenizer(
            sentence,
            return_tensors='pt',
            add_special_tokens=True
            )

        # 获取token列表
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # 找到目标词的位置
        target_positions = []
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                target_positions.append(i)

        if not target_positions:
            print(f"词 '{target_word}' 不在句子中")
            return None

        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(**encoded.to(self.device))
            hidden_states = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

        # 提取目标词的表示
        word_embeddings = []
        for pos in target_positions:
            word_embeddings.append(hidden_states[pos].cpu().numpy())

        return {
            'tokens': tokens,
            'positions': target_positions,
            'embeddings': word_embeddings
            }

    def visualize_sentence_embeddings(
            self, sentences: List[str],
            labels: List[str] = None,
            pooling_strategy='cls'
            ):
        """
        可视化句子嵌入
        """
        # 编码句子
        embeddings = self.encode_sentences(sentences, pooling_strategy)

        # 降维
        if len(sentences) > 50:
            # 先用PCA降到50维
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)

        # t-SNE降到2维
        perplexity = min(30, len(sentences) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # 绘图
        plt.figure(figsize=(12, 8))

        if labels is None:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
            # 添加句子片段作为标签
            for i, sent in enumerate(sentences):
                plt.annotate(
                    sent[:30] + "..." if len(sent) > 30 else sent,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8, alpha=0.7
                    )
        else:
            # 使用提供的标签着色
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.arange(len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = [l == label for l in labels]
                points = embeddings_2d[mask]
                plt.scatter(
                    points[:, 0], points[:, 1],
                    c=[color], label=label, alpha=0.6
                    )

            plt.legend()

        plt.title(f'Sentence Embeddings Visualization (pooling={pooling_strategy})')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        plt.show()

    def semantic_similarity_demo(self):
        """
        语义相似度演示
        """
        print("\n=== 语义相似度演示 ===\n")

        # 测试句子对
        sentence_pairs = [
            ("The car is parked in the garage.", "The automobile is in the parking lot."),
            ("I love eating pizza.", "Pizza is my favorite food."),
            ("The weather is nice today.", "It's a beautiful day."),
            ("He plays football.", "She dances ballet."),
            ("The stock market crashed.", "The economy is in recession."),
            ("Python is a programming language.", "Java is used for coding."),
            ]

        for sent1, sent2 in sentence_pairs:
            sim = self.compute_similarity([sent1], [sent2])[0][0]
            print(f"相似度: {sim:.3f}")
            print(f"  句子1: {sent1}")
            print(f"  句子2: {sent2}\n")


# 使用示例
def main():
    # 初始化BERT编码器（使用较小的模型以加快速度）
    # 如果需要中文支持，使用 'bert-base-chinese'
    encoder = BertSentenceEncoder('bert-base-uncased')

    # 1. 基本句子编码
    print("\n=== 基本句子编码示例 ===")
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Machine learning is fascinating.",
        "Deep learning transforms artificial intelligence.",
        "I enjoy reading books.",
        "Books are a source of knowledge.",
        ]

    # 使用不同的池化策略
    for strategy in ['cls', 'mean', 'max']:
        print(f"\n池化策略: {strategy}")
        embeddings = encoder.encode_sentences(test_sentences, pooling_strategy=strategy)
        print(f"句子向量形状: {embeddings.shape}")

    # 2. 展示上下文感知能力
    encoder.demonstrate_context_awareness()

    # 3. 语义相似度演示
    encoder.semantic_similarity_demo()

    # 4. 可视化句子向量
    print("\n=== 句子向量可视化 ===")

    # 准备不同类别的句子
    categories = {
        'Technology': [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning algorithms process big data.",
            "Neural networks mimic human brain functions.",
            "Python is popular for data science.",
            ],
        'Sports': [
            "The football team won the championship.",
            "Basketball players need good stamina.",
            "Tennis requires precision and agility.",
            "Swimming is an excellent exercise.",
            ],
        'Food': [
            "Italian pasta is delicious.",
            "Sushi is a Japanese delicacy.",
            "Fresh vegetables are healthy.",
            "Chocolate cake is a sweet dessert.",
            ],
        'Nature': [
            "The mountain peaks are covered with snow.",
            "Ocean waves crash against the shore.",
            "Forests provide oxygen for the planet.",
            "Desert landscapes are surprisingly beautiful.",
            ]
        }

    all_sentences = []
    all_labels = []
    for category, sents in categories.items():
        all_sentences.extend(sents)
        all_labels.extend([category] * len(sents))

    encoder.visualize_sentence_embeddings(all_sentences, all_labels)

    # 5. 特定词在不同上下文中的表示
    print("\n=== 词在上下文中的表示 ===")
    word_contexts = [
        ("The bank of the river is steep.", "bank"),
        ("I deposited money in the bank.", "bank"),
        ("She plays bass in a band.", "bass"),
        ("I caught a large bass while fishing.", "bass"),
        ]

    context_embeddings = []
    context_labels = []

    for sentence, target_word in word_contexts:
        result = encoder.extract_word_embeddings_in_context(sentence, target_word)
        if result:
            print(f"\n句子: {sentence}")
            print(f"目标词: {target_word}")
            print(f"Tokens: {result['tokens']}")
            print(f"位置: {result['positions']}")

            # 收集用于可视化
            for emb in result['embeddings']:
                context_embeddings.append(emb)
                context_labels.append(f"{target_word} in '{sentence[:20]}...'")


if __name__ == "__main__":
    main()
