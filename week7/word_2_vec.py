import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE


class Word2Vec:
    def __init__(
            self, sentences, embedding_size=100, window_size=2,
            min_count=5, negative_samples=5, epochs=5, lr=0.01
            ):
        """
        Word2Vec模型实现

        参数:
            sentences: 句子列表，每个句子是词的列表
            embedding_size: 词向量维度
            window_size: 上下文窗口大小
            min_count: 最小词频阈值
            negative_samples: 负采样数量
            epochs: 训练轮数
            lr: 学习率
        """
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.epochs = epochs
        self.lr = lr

        # 构建词汇表
        self.build_vocab(sentences, min_count)
        # 准备训练数据
        self.prepare_training_data(sentences)
        # 初始化模型
        self.init_model()

    def build_vocab(self, sentences, min_count):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)

        # 过滤低频词
        self.vocab = {word: idx for idx, (word, count) in
                      enumerate(word_counts.items()) if count >= min_count}
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 计算词频用于负采样
        word_freqs = np.array(
            [word_counts[self.idx2word[i]]
             for i in range(self.vocab_size)]
            )
        # 使用 3/4 次幂平滑词频分布
        word_freqs = word_freqs ** 0.75
        self.word_probs = word_freqs / np.sum(word_freqs)

        print(f"词汇表大小: {self.vocab_size}")

    def prepare_training_data(self, sentences):
        """准备训练数据（目标词-上下文词对）"""
        self.training_data = []

        for sentence in sentences:
            # 将词转换为索引
            indices = [self.vocab[word] for word in sentence
                       if word in self.vocab]

            # 生成训练样本
            for center_pos, center_idx in enumerate(indices):
                # 获取上下文窗口
                context_indices = []
                for offset in range(-self.window_size, self.window_size + 1):
                    context_pos = center_pos + offset
                    if (offset != 0 and 0 <= context_pos < len(indices)):
                        context_indices.append(indices[context_pos])

                if context_indices:
                    self.training_data.append((center_idx, context_indices))

        print(f"训练样本数: {len(self.training_data)}")

    def init_model(self):
        """初始化模型参数"""
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(
            self.vocab_size,
            self.embedding_size
            )
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(
            self.vocab_size,
            self.embedding_size
            )

        # Xavier初始化
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

        # 优化器
        self.optimizer = optim.Adam(
            [
                self.center_embeddings.weight,
                self.context_embeddings.weight
                ], lr=self.lr
            )

    def negative_sampling(self, positive_idx, num_samples):
        """负采样"""
        negative_indices = []
        while len(negative_indices) < num_samples:
            neg_idx = np.random.choice(self.vocab_size, p=self.word_probs)
            if neg_idx != positive_idx:
                negative_indices.append(neg_idx)
        return negative_indices

    def train_step(self, center_idx, context_indices):
        """单步训练（Skip-gram with Negative Sampling）"""
        self.optimizer.zero_grad()

        center_idx = torch.tensor([center_idx])
        center_vec = self.center_embeddings(center_idx)

        loss = 0.0

        for context_idx in context_indices:
            # 正样本
            context_idx_tensor = torch.tensor([context_idx])
            context_vec = self.context_embeddings(context_idx_tensor)

            # 计算正样本得分
            pos_score = torch.sum(center_vec * context_vec)
            pos_loss = -torch.log(torch.sigmoid(pos_score))
            loss += pos_loss

            # 负样本
            neg_indices = self.negative_sampling(
                context_idx,
                self.negative_samples
                )
            neg_indices_tensor = torch.tensor(neg_indices)
            neg_vecs = self.context_embeddings(neg_indices_tensor)

            # 计算负样本得分
            neg_scores = torch.matmul(neg_vecs, center_vec.T).squeeze()
            neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_scores)))
            loss += neg_loss

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """训练模型"""
        print("开始训练...")
        for epoch in range(self.epochs):
            total_loss = 0
            np.random.shuffle(self.training_data)

            for i, (center_idx, context_indices) in enumerate(self.training_data):
                loss = self.train_step(center_idx, context_indices)
                total_loss += loss

                if i % 1000 == 0 and i > 0:
                    avg_loss = total_loss / i
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}, "
                        f"Step {i}/{len(self.training_data)}, "
                        f"Loss: {avg_loss:.4f}"
                        )

            print(f"Epoch {epoch + 1} 完成, 平均损失: {total_loss / len(self.training_data):.4f}")

    def get_embedding(self, word):
        """获取词向量"""
        if word not in self.vocab:
            return None
        idx = self.vocab[word]
        # 使用中心词嵌入作为最终的词向量
        return self.center_embeddings.weight[idx].detach().numpy()

    def find_similar_words(self, word, top_k=10):
        """查找相似词"""
        if word not in self.vocab:
            return []

        word_vec = self.get_embedding(word)
        similarities = []

        for other_word in self.vocab:
            if other_word != word:
                other_vec = self.get_embedding(other_word)
                # 计算余弦相似度
                sim = np.dot(word_vec, other_vec) / (
                        np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                similarities.append((other_word, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def visualize_embeddings(self, words=None, max_words=100):
        """可视化词向量"""
        if words is None:
            # 选择高频词
            words = list(self.vocab.keys())[:max_words]

        # 获取词向量
        embeddings = []
        labels = []
        for word in words:
            if word in self.vocab:
                embeddings.append(self.get_embedding(word))
                labels.append(word)

        embeddings = np.array(embeddings)

        # 检查样本数量
        n_samples = len(embeddings)
        if n_samples < 2:
            print("词向量数量太少，无法可视化")
            return

        # 使用t-SNE降维到2D
        # 动态设置perplexity，确保它小于样本数
        perplexity = min(30, n_samples - 1)  # perplexity必须小于n_samples

        print(f"可视化 {n_samples} 个词向量，perplexity={perplexity}")

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # 绘图
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

        # 添加词标签
        for i, label in enumerate(labels):
            plt.annotate(
                label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8, alpha=0.7
                )

        plt.title(f'Word Embeddings Visualization (t-SNE, {n_samples} words)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.show()


# 示例：训练和测试
def preprocess_text(text):
    """文本预处理"""
    # 转换为小写
    text = text.lower()
    # 分词（简单按空格分割）
    words = re.findall(r'\b\w+\b', text)
    return words


if __name__ == "__main__":

    # 准备示例数据
    sample_text = """
    The king ruled the kingdom with wisdom. The queen supported the king.
    The prince learned from the king. The princess studied with the queen.
    The man worked in the fields. The woman cared for the children.
    The boy played with the dog. The girl fed the cat.
    Dogs are loyal animals. Cats are independent pets.
    Apples grow on trees. Oranges are citrus fruits.
    Red roses are beautiful. Blue skies are peaceful.
    Programming requires logic. Python simplifies coding.
    Machine learning predicts patterns. Deep learning mimics brains.
    Natural language is complex. Word embeddings capture meaning.
    Vectors represent concepts. Mathematics underlies algorithms.
    Computers process information. Software runs on hardware.
    Data drives decisions. Analysis reveals insights.
    Training improves models. Testing validates results.
    The sun shines bright. The moon glows at night.
    Summer brings warmth. Winter brings snow.
    Music soothes souls. Art inspires creativity.
    Books contain knowledge. Libraries store books.
    Teachers educate students. Students learn lessons.
    """

    # 将文本分割成句子
    sentences = [preprocess_text(sent) for sent in sample_text.split('.') if sent.strip()]

    # 训练Word2Vec模型
    model = Word2Vec(
        sentences,
        embedding_size=100,
        window_size=3,
        min_count=1,
        negative_samples=5,
        epochs=30,
        lr=0.01
        )

    model.train()

    # 测试语义关系
    print("\n=== 语义关系测试 ===")
    test_words = ['king', 'queen', 'man', 'woman', 'python', 'learning']

    for word in test_words:
        print(f"\n'{word}' 的相似词:")
        similar_words = model.find_similar_words(word, top_k=5)
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")

    # 词向量算术测试
    print("\n=== 词向量算术 ===")


    def word_analogy(model, word1, word2, word3):
        """词类比: word1 - word2 + word3 ≈ ?"""
        vec1 = model.get_embedding(word1)
        vec2 = model.get_embedding(word2)
        vec3 = model.get_embedding(word3)

        if vec1 is None or vec2 is None or vec3 is None:
            return None

        # 计算 word1 - word2 + word3
        result_vec = vec1 - vec2 + vec3

        # 找最相似的词
        best_word = None
        best_sim = -1

        for word in model.vocab:
            if word not in [word1, word2, word3]:
                vec = model.get_embedding(word)
                sim = np.dot(result_vec, vec) / (
                        np.linalg.norm(result_vec) * np.linalg.norm(vec))
                if sim > best_sim:
                    best_sim = sim
                    best_word = word

        return best_word, best_sim


    # 测试词类比
    if 'king' in model.vocab and 'man' in model.vocab and 'woman' in model.vocab:
        result = word_analogy(model, 'king', 'man', 'woman')
        if result:
            print(f"king - man + woman ≈ {result[0]} (相似度: {result[1]:.3f})")

    # 可视化词向量
    print("\n生成词向量可视化...")
    model.visualize_embeddings(max_words=30)
