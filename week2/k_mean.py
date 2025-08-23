# --- 第1步：准备工具 ---
import os
import warnings

import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer  # 你提到的主角登场！
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler  # 我们需要一个新的“裁判”工具

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # --- 第2步：加载新格式的数据 ---
    # 我根据你给的格式，创建一个新的示例文件 'user_profiles_v2.csv'
    file_path = '../doc/user_profiles_en.csv'
    if not os.path.exists(file_path):
        print(f"没找到 '{file_path}'，我先帮你创建一个示例数据...")
        exit()
    else:
        df_original = pd.read_csv(file_path)
        print("成功加载新版用户数据！原始数据长这样：")
        print(df_original.head())

    print("\n" + "=" * 50 + "\n")

    # --- 第3步：用 BGE-M3 模型进行特征“深度理解” ---
    print("正在加载 BAAI/bge-m3 模型，这可能需要一点时间，请稍等...")
    # 加载我们强大的语言模型
    model = SentenceTransformer('BAAI/bge-m3')
    print("模型加载完毕！")

    # 我们要处理的文字特征列
    text_features = ['gender', 'city', 'consumption_level']
    encoded_features_list = []

    print("开始用模型'阅读'并转换文字特征...")
    for feature in text_features:
        # 把一列文字，比如['Male', 'Female', 'Male'...]，交给模型去“阅读”
        embeddings = model.encode(df_original[feature].tolist(), show_progress_bar=True)

        # 把模型输出的一大堆数字（嵌入向量）转换成pandas的DataFrame格式
        df_encoded = pd.DataFrame(embeddings, index=df_original.index)
        df_encoded.columns = [f'{feature}_emb_{i}' for i in range(embeddings.shape[1])]
        encoded_features_list.append(df_encoded)
        print(f"'{feature}' 特征已成功转换为向量。")

    # 把所有转换好的文字特征和原始的数字特征拼接在一起
    df_numeric_features = df_original[['age', 'active_days']]
    df_for_clustering = pd.concat([df_numeric_features] + encoded_features_list, axis=1)

    print("\n" + "所有特征都已转换为纯数字，准备进行聚类。处理后的数据维度为:", df_for_clustering.shape)
    print("看一眼处理后的数据（只显示前5列）：")
    print(df_for_clustering.iloc[:, :5].head())

    print("\n" + "=" * 50 + "\n")

    # --- 第4步：特征“公平化”处理 (标准化) ---
    # 这一步非常关键！你想想，'age'的数值范围是几十，而模型嵌入的每个数值都在-1到1之间。
    # 如果直接聚类，'age'的影响力会比其他所有特征加起来都大，这不公平。
    # StandardScaler就像一个裁判，把所有特征拉到同一个起跑线上，让它们公平竞争。
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_for_clustering)

    print("已对所有特征进行'公平化'处理（标准化），确保所有特征的'嗓门'一样大。")

    print("\n" + "=" * 50 + "\n")

    # --- 第5步：执行K-Means聚类 ---
    # 和上次一样，还是那个熟悉的分堆过程
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    # 注意！我们是在“公平化”之后的数据上进行聚类的
    kmeans.fit(df_scaled)

    print(f"聚类完成！已经成功将用户分成了 {k} 个群体。")

    print("\n" + "=" * 50 + "\n")
    # --- 新增步骤 5.5: 数据降维与可视化 ---
    print("--- 正在使用'PCA相机'为高维数据拍摄'二维照片'以便于观察 ---")
    # 1. 降维到3D (和之前一样)
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df_scaled)

    # 2. 准备要在悬停框中显示的数据
    # 我们把原始数据中需要显示的信息，按顺序整理好
    custom_data = df_original[['user_id', 'gender', 'age', 'city', 'consumption_level', 'active_days']].to_numpy()

    # 3. 定义悬停框的模板
    # 这里有点像在写一个迷你的网页，告诉Plotly鼠标放上去时要显示什么内容
    # %{customdata[i]} 会从我们上面准备的 custom_data 中取出第 i 列的数据
    hover_template = (
            "<b>User ID: %{customdata[0]}</b><br><br>" +
            "Gender: %{customdata[1]}<br>" +
            "Age: %{customdata[2]}<br>" +
            "City: %{customdata[3]}<br>" +
            "Consumption: %{customdata[4]}<br>" +
            "Active Days: %{customdata[5]}<br>" +
            "<extra></extra>"  # <extra>标签让plotly不要显示多余的trace名字
    )

    # 4. 创建3D散点图对象
    scatter_trace = go.Scatter3d(
        x=df_pca[:, 0],
        y=df_pca[:, 1],
        z=df_pca[:, 2],
        mode='markers',  # 只画点
        marker=dict(
            size=5,
            color=kmeans.labels_,  # 用聚类结果上色
            colorscale='Viridis',  # 颜色主题
            opacity=0.8
            ),
        customdata=custom_data,  # 把我们的详细信息“绑定”到每个点上
        hovertemplate=hover_template  # 应用我们设计的悬停模板
        )

    # 5. 把中心点也画出来 (和之前一样，但用Plotly的语法)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroids_trace = go.Scatter3d(
        x=centroids_pca[:, 0],
        y=centroids_pca[:, 1],
        z=centroids_pca[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='x'  # 用 'X' 标记
            ),
        name='Centroids'  # 在图例中显示的名字
        )

    # 6. 组合图像并设置布局
    fig = go.Figure(data=[scatter_trace, centroids_trace])
    fig.update_layout(
        title='Interactive 3D User Clusters (Hover for Details)',
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
            ),
        margin=dict(l=0, r=0, b=0, t=40)  # 调整边界
        )

    print("Plot generated! It will open in your web browser.")
    # 7. 显示图像！它会在你的浏览器里打开一个可交互的网页
    fig.show()

    print("\n" + "=" * 50 + "\n")
    # --- 第6步：寻找群体代表（用户画像） ---
    # 这里的逻辑和上次完全一样，但要记住，距离计算也要在“公平化”后的数据上进行
    # 1. 获取每个群体的中心点（它也在“公平化”后的空间里）
    centroids_scaled = kmeans.cluster_centers_

    # 2. 在“公平化”后的数据里，找到离每个中心点最近的真实用户的“行号”
    closest_user_indices, _ = pairwise_distances_argmin_min(centroids_scaled, df_scaled)

    # 3. 根据找到的“行号”，从我们最最原始的用户名单里把这几位“代言人”请出来
    representative_users = df_original.iloc[closest_user_indices]

    print("--- 找到了3个群体的'形象代言人'！他们的数据如下：---")
    for i in range(len(representative_users)):
        print(f"\n--- 群体 {i} 代表用户 ---")
        print(representative_users.iloc[i].to_frame().T)

    print("\n" + "=" * 50 + "\n")

    # --- 第7步：分析与解读 ---
    print("--- 基于'代言人'数据，我们来猜猜这3个群体分别是什么样的人：---")
    # （这里的分析逻辑需要根据你实际跑出的结果进行调整）
    for i in range(len(representative_users)):
        user = representative_users.iloc[i]
        analysis = f"群体 {i} 的代表是用户 {user['user_id']} ({user['gender']}, {user['age']}岁, 来自{user['city']})。 "

        # 基于消费和活跃度进行简单画像分析
        consumption = user['consumption_level']
        active_days = user['active_days']

        if consumption == 'High' and active_days >= 20:
            analysis += f"画像是【高价值忠实用户】。消费力强({consumption})，且几乎天天都来({active_days}天)。他们是我们的金主爸爸，必须得好好伺候着！"
        elif consumption == 'Low' and active_days <= 7:
            analysis += (f"画像是【待激活的新手用户】。消费水平({consumption})和活跃度("
                         f"{active_days}天)都偏低。这可能是刚注册没多久的用户，需要我们多用些新手引导和福利活动来'勾引'他们留下来。")
        elif consumption == 'Medium' and active_days > 10:
            analysis += (f"画像是【稳健的腰部用户】。有稳定的活跃度({active_days}天)和中等的消费("
                         f"{consumption})。他们是平台的中流砥柱，可以通过一些促销活动或者积分体系，鼓励他们向高价值用户迈进。")
        else:
            analysis += (f"画像是【潜力观望用户】。可能有一定的消费意愿({consumption})或活跃度("
                         f"{active_days}天)，但表现还不够突出。需要进一步分析他们的行为，看看是什么阻碍了他们成为更核心的用户。")

        print(analysis + "\n")
