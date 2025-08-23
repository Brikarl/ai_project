# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
# ### 变化点1：导入的库不同了 ###
# 我们不再用 FlagEmbedding，而是用更通用的 SentenceTransformer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    # --- 第1步: 数据准备 (这部分完全不变) ---
    print("--- 1. 准备数据 ---")
    positive_samples = [
        "这家的服务态度简直了，让人心里暖洋洋的。", "刚开始没抱多大希望，结果效果出奇的好！",
        "虽然等了很久，但吃到嘴里那一刻觉得一切都值了。", "这个产品设计得太巧妙了，解决了我一直以来的痛点。",
        "客服小哥非常有耐心，一步步教我怎么操作，必须点赞。", "价格不贵，质量却一点也不含糊。",
        "电影的后半段反转再反转，看得我大呼过瘾。", "朋友推荐过来的，果然名不虚传。",
        "包装很用心，连个小角都没磕碰到。", "不得不说，这是我今年买过最满意的一件东西。",
        "本来以为会很复杂，没想到上手这么快。", "他们的售后响应速度很快，问题马上就解决了。",
        ]
    negative_samples = [
        "等了一个多小时，上来一杯凉白开，体验感极差。", "图片仅供参考，实物和照片简直是两个东西。",
        "说明书跟天书一样，研究了半天也没看懂。", "用了一次就坏了，这质量真是一言难尽。",
        "说好的24小时客服，半夜发消息根本没人回。", "到处都是广告弹窗，用起来太烦人了。",
        "一股廉价的塑料味，闻着就头疼。", "物流太慢了，同城的快递走了三天。",
        "哦，太棒了，刚过保修期它就坏了。", "除了价格便宜，我找不出任何优点。",
        "宣传的功能很强大，实际好多都用不了。", "感觉自己被当猴耍了，再也不会来了。",
        ]
    all_sentences = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    print(f"数据准备完毕！总共有 {len(all_sentences)} 条句子。")

    # --- 第2步: 特征提取 (这里的工具换了，但做的事情一样) ---
    print("\n--- 2. 特征提取 ---")
    # ### 变化点2：加载模型的方式更直接 ###
    # 第一次运行同样会自动下载模型文件。
    print("正在加载SentenceTransformer模型，可能需要几分钟...")
    model = SentenceTransformer('BAAI/bge-m3')
    print("模型加载完毕！")

    print("正在把所有句子转换成语义向量(Embedding)...")
    # ### 变化点3：转换向量的代码变得超级简单！ ###
    # model.encode() 直接就返回了我们想要的Numpy数组，不用再从字典里取了。
    embeddings = model.encode(all_sentences)

    # 特征X就是我们的向量，标签y还是老样子
    X = embeddings
    y = np.array(labels)
    print(f"转换完成！我们得到了一个 {X.shape} 的特征矩阵X。")

    # --- 第3步: 模型训练 (这部分完全不变，因为“画线工人”还是那个工人) ---
    print("\n--- 3. 模型训练 ---")
    print("开始训练逻辑回归分类器...")
    classifier = LogisticRegression(C=1.0, max_iter=1000)
    classifier.fit(X, y)
    print("分类器训练完成！")

    # --- 第4步: 准备可视化数据 (PCA降维) ---
    # 这个步骤纯粹是为了画图，不影响主力模型的性能
    print("\n--- 4. 准备可视化数据 (降维到3D) ---")
    pca = PCA(n_components=3)
    # 用PCA把高维数据“拍成”3D照片
    X_3d = pca.fit_transform(X)
    print("数据已降至3D，用于绘图。")

    # --- 第4步: 效果评估 (这里的向量转换方式也要跟着变) ---
    print("\n--- 4. 效果评估 ---")
    test_sentences = [
        "这家店的氛围感拉满了，下次还来。",
        "排队两小时，吃饭五分钟，不会再爱了。",
        "也就那样吧，不好不坏。",
        "强烈推荐！性价比之王！"
        ]

    # ### 变化点4：测试句子的转换方式也同步更新 ###
    print("正在把测试句子转换成向量...")
    test_X = model.encode(test_sentences)

    print("正在用训练好的分类器进行预测...")
    predictions = classifier.predict(test_X)
    probabilities = classifier.predict_proba(test_X)

    # 结果展示部分完全不变
    print("\n--- 预测结果 ---")
    for i, sentence in enumerate(test_sentences):
        label = "积极" if predictions[i] == 1 else "消极"
        prob_positive = probabilities[i][1]
        prob_negative = probabilities[i][0]

        print(f"\n句子: '{sentence}'")
        print(f"  -> 预测结果: 【{label}】")
        print(f"  -> 模型判断的信心: (积极概率: {prob_positive:.2%}, 消极概率: {prob_negative:.2%})")

    # 为了画图，把测试句子也降到3D
    test_X_3d = pca.transform(test_X)

    # --- 开始画图 ---
    print("\n正在生成三维可视化图形...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 画训练数据的点
    positive_points = X_3d[y == 1]
    negative_points = X_3d[y == 0]
    ax.scatter(
        positive_points[:, 0], positive_points[:, 1], positive_points[:, 2], c='blue', marker='o', s=50,
        label='Positive Samples'
        )
    ax.scatter(
        negative_points[:, 0], negative_points[:, 1], negative_points[:, 2], c='red', marker='x', s=50,
        label='Negative Samples'
        )

    # 画测试数据的点，并用真实预测结果标注
    ax.scatter(test_X_3d[:, 0], test_X_3d[:, 1], test_X_3d[:, 2], c='green', marker='*', s=200, label='Test Samples')
    for i, txt in enumerate(test_sentences):
        # 注意！这里的标签用的是 real_predictions
        label_text = "Positive" if predictions[i] == 1 else "Negative"
        ax.text(test_X_3d[i, 0], test_X_3d[i, 1], test_X_3d[i, 2], f"  Prediction:{label_text}", color='black')

    # --- 为了画出分界平面，我们临时在3D数据上训练一个“可视化辅助模型” ---
    # 这个模型存在的唯一意义，就是为了画出下面这个平面
    classifier_3d_for_viz = LogisticRegression()
    classifier_3d_for_viz.fit(X_3d, y)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    coef = classifier_3d_for_viz.coef_[0]
    intercept = classifier_3d_for_viz.intercept_
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', label='Decision Boundary')

    ax.set_xlabel('Main Component 1')
    ax.set_ylabel('Main Component 2')
    ax.set_zlabel('Main Component 3')
    ax.set_title('3D Visualization of Sentiment Analysis', fontsize=16)
    ax.legend()
    plt.show()
