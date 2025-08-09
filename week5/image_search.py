import os
import warnings
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
from typing import List, Tuple, Dict

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CLIPImageRetrieval:
    """基于CLIP的图像检索系统"""

    def __init__(self, model_name='ViT-B/32'):
        """
        初始化CLIP模型

        Args:
            model_name: CLIP模型名称，可选: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50'等
        """
        print(f"加载CLIP模型: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # 存储图像特征和路径
        self.image_features = None
        self.image_paths = []

    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """
        提取单张图片的特征向量

        Args:
            image_path: 图片路径

        Returns:
            归一化的图像特征向量
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # 归一化特征向量
            image_features = F.normalize(image_features, p=2, dim=1)

        return image_features

    def build_index(self, image_dir: str, save_index: bool = True, index_path: str = 'clip_index.pkl'):
        """
        构建图片索引库

        Args:
            image_dir: 包含图片的目录
            save_index: 是否保存索引到文件
            index_path: 索引保存路径
        """
        print(f"正在构建图片索引库...")

        # 收集所有图片路径
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

        if not self.image_paths:
            print("未找到图片！")
            return

        print(f"找到 {len(self.image_paths)} 张图片")

        # 提取所有图片的特征
        features_list = []
        for image_path in tqdm(self.image_paths, desc="提取特征"):
            try:
                features = self.extract_image_features(image_path)
                features_list.append(features.cpu())
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
                self.image_paths.remove(image_path)

        # 合并所有特征
        self.image_features = torch.cat(features_list, dim=0)

        # 保存索引
        if save_index:
            with open(index_path, 'wb') as f:
                pickle.dump(
                    {
                        'features': self.image_features,
                        'paths': self.image_paths
                        }, f
                    )
            print(f"索引已保存到: {index_path}")

    def load_index(self, index_path: str):
        """加载预构建的索引"""
        print(f"加载索引: {index_path}")
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.image_features = data['features']
            self.image_paths = data['paths']
        print(f"加载了 {len(self.image_paths)} 张图片的索引")

    def search_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        搜索相似图片

        Args:
            query_image_path: 查询图片路径
            top_k: 返回前K个最相似的结果

        Returns:
            [(图片路径, 相似度分数), ...]
        """
        if self.image_features is None:
            raise ValueError("请先构建或加载索引库")

        # 提取查询图片特征
        query_features = self.extract_image_features(query_image_path)

        # 计算余弦相似度
        similarities = torch.matmul(query_features, self.image_features.T).squeeze()

        # 获取top-k结果
        top_k_similarities, top_k_indices = similarities.topk(min(top_k, len(self.image_paths)))

        # 返回结果
        results = []
        for idx, sim in zip(top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()):
            results.append((self.image_paths[idx], float(sim)))

        return results

    def visualize_search_results(
            self, query_image_path: str, results: List[Tuple[str, float]],
            save_path: str = None
            ):
        """
        可视化搜索结果

        Args:
            query_image_path: 查询图片路径
            results: 搜索结果
            save_path: 保存图片路径（可选）
        """
        n_results = len(results) + 1
        fig = plt.figure(figsize=(15, 5))

        # 显示查询图片
        plt.subplot(1, n_results, 1)
        query_img = Image.open(query_image_path)
        plt.imshow(query_img)
        plt.title('Query Image', fontsize=12)
        plt.axis('off')

        # 显示搜索结果
        for i, (img_path, similarity) in enumerate(results, 2):
            plt.subplot(1, n_results, i)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f'Sim: {similarity:.3f}', fontsize=10)
            plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")

        plt.show()


class CLIPImageRetrievalAdvanced(CLIPImageRetrieval):
    """增强版CLIP图像检索系统，支持文本搜索和批量操作"""

    def __init__(self, model_name='ViT-B/32'):
        super().__init__(model_name)
        self.text_features = None
        self.text_descriptions = []

    def search_by_text(self, text_query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        使用文本搜索相似图片

        Args:
            text_query: 文本查询
            top_k: 返回前K个结果

        Returns:
            [(图片路径, 相似度分数), ...]
        """
        if self.image_features is None:
            raise ValueError("请先构建或加载索引库")

        # 编码文本
        text_tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)

        # 计算相似度
        similarities = torch.matmul(text_features, self.image_features.T).squeeze()

        # 获取top-k结果
        top_k_similarities, top_k_indices = similarities.topk(min(top_k, len(self.image_paths)))

        results = []
        for idx, sim in zip(top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()):
            results.append((self.image_paths[idx], float(sim)))

        return results

    def batch_search(self, query_images: List[str], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        批量搜索多张图片

        Args:
            query_images: 查询图片路径列表
            top_k: 每张图片返回的结果数

        Returns:
            {查询图片路径: [(结果图片路径, 相似度), ...], ...}
        """
        results = {}
        for query_path in tqdm(query_images, desc="批量搜索"):
            try:
                results[query_path] = self.search_similar_images(query_path, top_k)
            except Exception as e:
                print(f"搜索 {query_path} 时出错: {e}")
                results[query_path] = []

        return results

    def find_duplicates(self, threshold: float = 0.95) -> List[List[str]]:
        """
        查找重复或高度相似的图片

        Args:
            threshold: 相似度阈值

        Returns:
            重复图片组列表
        """
        if self.image_features is None:
            raise ValueError("请先构建或加载索引库")

        # 计算所有图片之间的相似度矩阵
        similarities = torch.matmul(self.image_features, self.image_features.T)

        # 查找重复组
        duplicates = []
        visited = set()

        for i in range(len(self.image_paths)):
            if i in visited:
                continue

            # 找到与当前图片相似度超过阈值的所有图片
            similar_indices = torch.where(similarities[i] > threshold)[0].cpu().numpy()

            if len(similar_indices) > 1:  # 包括自己
                duplicate_group = [self.image_paths[idx] for idx in similar_indices]
                duplicates.append(duplicate_group)
                visited.update(similar_indices)

        return duplicates


def demo_clip_retrieval():
    """演示CLIP图像检索系统的使用"""

    # 创建检索系统
    retriever = CLIPImageRetrievalAdvanced(model_name='ViT-B/32')

    # 构建索引
    dataset_dir = './dataset'  # 替换为你的图片目录
    if os.path.exists(dataset_dir):
        retriever.build_index(dataset_dir, save_index=True, index_path='clip_index.pkl')
    else:
        print("数据集目录不存在，创建示例目录结构...")
        os.makedirs(dataset_dir, exist_ok=True)
        return

    # 示例：图像搜索
    if len(retriever.image_paths) > 0:
        # 使用第一张图片作为查询
        query_image = retriever.image_paths[0]
        print(f"\n使用图片搜索: {query_image}")

        # 搜索相似图片
        results = retriever.search_similar_images(query_image, top_k=5)

        # 可视化结果
        retriever.visualize_search_results(query_image, results[1:])  # 排除自己

        # 文本搜索示例
        text_query = "a dog"
        print(f"\n使用文本搜索: '{text_query}'")
        text_results = retriever.search_by_text(text_query, top_k=5)

        # 显示文本搜索结果
        if text_results:
            fig = plt.figure(figsize=(15, 3))
            for i, (img_path, similarity) in enumerate(text_results, 1):
                plt.subplot(1, len(text_results), i)
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(f'Sim: {similarity:.3f}', fontsize=10)
                plt.axis('off')
            plt.suptitle(f'Text Query: "{text_query}"', fontsize=14)
            plt.tight_layout()
            plt.show()

        # 查找重复图片
        print("\n查找重复图片...")
        duplicates = retriever.find_duplicates(threshold=0.95)
        if duplicates:
            print(f"找到 {len(duplicates)} 组重复图片")
            for i, group in enumerate(duplicates, 1):
                print(f"组 {i}: {len(group)} 张相似图片")
        else:
            print("未找到重复图片")


def create_simple_gui():
    """创建简单的图像检索GUI（使用gradio）"""
    try:
        import gradio as gr

        # 创建检索系统
        retriever = CLIPImageRetrievalAdvanced()

        # 加载索引
        if os.path.exists('clip_index.pkl'):
            retriever.load_index('clip_index.pkl')
        else:
            print("请先运行 demo_clip_retrieval() 构建索引")
            return

        def search_images(query_image, query_text, search_mode, top_k):
            """搜索函数"""
            results = []

            if search_mode == "Image":
                if query_image is None:
                    return []
                # 临时保存上传的图片
                temp_path = "temp_query.jpg"
                query_image.save(temp_path)
                search_results = retriever.search_similar_images(temp_path, top_k=top_k)
                os.remove(temp_path)
            else:  # Text mode
                if not query_text:
                    return []
                search_results = retriever.search_by_text(query_text, top_k=top_k)

            # 准备返回的图片列表
            for img_path, similarity in search_results:
                img = Image.open(img_path)
                results.append((img, f"Similarity: {similarity:.3f}"))

            return results

        # 创建Gradio界面
        with gr.Blocks(title="CLIP Image Retrieval") as demo:
            gr.Markdown("# CLIP Image Retrieval System")

            with gr.Row():
                with gr.Column():
                    search_mode = gr.Radio(["Image", "Text"], value="Image", label="Search Mode")
                    query_image = gr.Image(type="pil", label="Query Image")
                    query_text = gr.Textbox(label="Query Text", placeholder="Enter text description...")
                    top_k = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
                    search_btn = gr.Button("Search", variant="primary")

                with gr.Column():
                    results_gallery = gr.Gallery(label="Search Results", columns=3, height="auto")

            search_btn.click(
                search_images,
                inputs=[query_image, query_text, search_mode, top_k],
                outputs=results_gallery
                )

        demo.launch()

    except ImportError:
        print("Gradio未安装。请运行: pip install gradio")


if __name__ == "__main__":
    # 安装CLIP (如果还没有安装)
    # pip install git+https://github.com/openai/CLIP.git

    # 运行演示
    # demo_clip_retrieval()

    # 如果想要GUI界面，取消下面的注释
    create_simple_gui()
