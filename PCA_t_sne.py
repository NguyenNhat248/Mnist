import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import plotly.express as px
import shutil
import time
import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 



def ly_thuyet_PCA(): 
    st.header("📖 Tổng quan về PCA")

    # Giới thiệu PCA
    st.markdown("### 1️⃣ PCA là gì?")
    st.write(
        "PCA (Principal Component Analysis) là một phương pháp giảm chiều dữ liệu phổ biến. "
        "Mục tiêu của PCA là biến đổi tập dữ liệu có nhiều chiều thành không gian có ít chiều hơn, "
        "trong khi vẫn giữ lại phần lớn thông tin quan trọng."
    )

    # Ý tưởng của PCA
    st.markdown("### 2️⃣ Ý tưởng chính")
    st.write(
        "- **Giảm chiều dữ liệu**: PCA tìm ra các hướng chứa nhiều thông tin nhất để biểu diễn dữ liệu.\n"
        "- **Loại bỏ nhiễu**: Những thành phần có phương sai nhỏ sẽ bị loại bỏ vì chúng ít đóng góp vào sự thay đổi của dữ liệu.\n"
        "- **Trích xuất đặc trưng quan trọng**: Giúp nhận diện cấu trúc dữ liệu tốt hơn."
    )

    # Các bước thực hiện PCA
    st.markdown("### 3️⃣ Các bước thực hiện PCA")
    
    # Bước 1: Tính trung bình
    st.markdown("#### 🔹 Bước 1: Tính giá trị trung bình")
    st.latex(r"\mu = \frac{1}{n} \sum_{i=1}^{n} x_i")
    st.write("Dịch chuyển dữ liệu sao cho trung tâm dữ liệu nằm tại gốc tọa độ.")

    # Bước 2: Tính ma trận hiệp phương sai
    st.markdown("#### 🔹 Bước 2: Tính ma trận hiệp phương sai")
    st.latex(r"C = \frac{1}{n} X^T X")
    st.write("Ma trận này giúp xác định mối quan hệ giữa các biến.")

    # Bước 3: Tính trị riêng và vector riêng
    st.markdown("#### 🔹 Bước 3: Tìm trị riêng và vector riêng")
    st.latex(r"C v = \lambda v")
    st.write(
        "Các vector riêng tương ứng với các hướng quan trọng nhất của dữ liệu, "
        "còn trị riêng thể hiện mức độ quan trọng của từng hướng."
    )

    # Bước 4: Chọn số chiều mới
    st.markdown("#### 🔹 Bước 4: Chọn số chiều chính")
    st.latex(r"U_K = [v_1, v_2, \dots, v_K]")
    st.write("Chọn số lượng vector riêng lớn nhất để giữ lại nhiều thông tin nhất.")

    # Bước 5: Chiếu dữ liệu vào không gian mới
    st.markdown("#### 🔹 Bước 5: Chiếu dữ liệu vào không gian mới")
    st.latex(r"X_{new} = X U_K")
    st.write("Dữ liệu sau khi chiếu vào không gian mới sẽ có số chiều nhỏ hơn.")

    # Minh họa bằng biểu đồ
    st.markdown("### 4️⃣ Minh họa PCA bằng biểu đồ")
    
    num_samples = st.slider("Số lượng dữ liệu 🟢", 100, 1000, 300, step=50)
    num_features = st.slider("Số chiều ban đầu 🔵", 3, 10, 3)
    num_clusters = st.slider("Số cụm 🔴", 2, 5, 3)
    
    max_components = max(2, num_features)
    n_components = st.slider("Số thành phần PCA 🟣", 2, max_components, min(2, max_components))

    if st.button("📊 Thực hiện PCA"):
        X, y = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, random_state=42)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Áp dụng PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        scatter1 = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="viridis", alpha=0.6)
        axes[0].set_title("Dữ liệu gốc")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        fig.colorbar(scatter1, ax=axes[0])

        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1] if n_components > 1 else np.zeros_like(X_pca[:, 0]), c=y, cmap="viridis", alpha=0.6)
        axes[1].set_title(f"Dữ liệu sau PCA ({n_components}D)")
        axes[1].set_xlabel("PC 1")
        if n_components > 1:
            axes[1].set_ylabel("PC 2")
        else:
            axes[1].set_yticks([])
        fig.colorbar(scatter2, ax=axes[1])

        st.pyplot(fig)

def ly_thuyet_tSne():
# Tiêu đề chính
    st.title("🔢 Khám phá t-SNE")

    st.write("""
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** là một phương pháp **giảm chiều dữ liệu**, 
    cho phép biểu diễn dữ liệu nhiều chiều dưới dạng **2D hoặc 3D** mà vẫn bảo toàn mối quan hệ gần gũi giữa các điểm.
    """)

    # Ý tưởng chính
    st.header("🔽 Cách hoạt động")

    st.markdown("""
    - **Mục tiêu**: Giảm số chiều nhưng vẫn duy trì sự tương đồng giữa các điểm.
    - **Quy trình**:
        1. **Xác suất cao chiều**: Sử dụng phân phối Gaussian để đo độ tương đồng.
        2. **Xác suất thấp chiều**: Dùng phân phối t-Student để hạn chế ảnh hưởng của các giá trị ngoại lai.
        3. **Tối ưu hóa**: Dựa trên KL-Divergence để làm cho hai phân phối này gần giống nhau nhất có thể.
    """)

    # Công thức
    st.header("📊 Một số công thức quan trọng")

    st.markdown("**Xác suất trong không gian gốc:**")
    st.latex(r"""
    p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma^2)}
    """)

    st.markdown("**Xác suất trong không gian nhúng:**")
    st.latex(r"""
    q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
    """)

    st.markdown("**Tối ưu hóa bằng KL-Divergence:**")
    st.latex(r"""
    KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
    """)

    st.success("✅ t-SNE giúp hiển thị dữ liệu đa chiều hiệu quả!")

    # Ứng dụng thực tế
    st.title("📉 Thử nghiệm t-SNE")

    # Chọn tham số
    num_samples = st.slider("Số điểm dữ liệu", 100, 1000, 300, step=50, key="num_samples")
    num_clusters = st.slider("Số cụm", 2, 5, 3, key="num_clusters")
    perplexity = st.slider("Perplexity", 5, 50, 30, key="perplexity")

    # Nút reset
    if st.button("🔄 Reset", key="reset_button"):
        st.rerun()

    # Chạy thuật toán
    if st.button("📊 Thực hiện", key="process_button"):
        st.write("### 🔹 Khởi tạo dữ liệu")
        X, y = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=3, random_state=42)
        st.write(f"✅ Dữ liệu gồm {num_samples} điểm và {num_clusters} cụm.")
        
        # Áp dụng t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        st.write("✅ t-SNE đã hoàn thành việc giảm chiều!")


def data(): 
    st.title("📚 Giới Thiệu Tập Dữ Liệu MNIST")
    
    st.markdown("""
    **MNIST (Modified National Institute of Standards and Technology)** là một trong những tập dữ liệu phổ biến nhất trong lĩnh vực học máy, đặc biệt là trong nhận dạng chữ số viết tay. Bộ dữ liệu này được thiết kế để hỗ trợ các thuật toán phân loại và nhận dạng mẫu. 
    
    ## 1. Cấu Trúc Dữ Liệu:
    Bộ dữ liệu MNIST gồm hai tập chính:
    
    - **Tập huấn luyện**: 60.000 mẫu dữ liệu.
    - **Tập kiểm tra**: 10.000 mẫu dữ liệu.
    
    Mỗi mẫu là một hình ảnh có kích thước 28x28 pixel, đại diện cho một chữ số từ 0 đến 9. Dữ liệu đã được tiền xử lý sẵn, bao gồm việc căn chỉnh hình ảnh và chuẩn hóa nền để hỗ trợ việc huấn luyện mô hình dễ dàng hơn.
    
    ## 2. Ứng Dụng Của MNIST:
    - **Phân loại chữ số viết tay**: Dự đoán số tương ứng với từng hình ảnh.
    - **Thử nghiệm thuật toán học máy**: Đánh giá hiệu suất của các mô hình khác nhau, từ các phương pháp truyền thống như KNN, SVM đến các mạng nơ-ron sâu.
    - **Tiền xử lý dữ liệu hình ảnh**: Giúp làm quen với các kỹ thuật chuẩn hóa và xử lý dữ liệu trước khi đưa vào mô hình.
    
    ## 3. Đặc Điểm Kỹ Thuật:
    Mỗi hình ảnh được biểu diễn dưới dạng ma trận 28x28 pixel, trong đó mỗi phần tử thể hiện mức độ sáng của pixel. Khi làm việc với tập dữ liệu này, các nhà nghiên cứu có thể thử nghiệm nhiều phương pháp khác nhau để cải thiện độ chính xác của mô hình phân loại.
    
    ## 4. Các Lĩnh Vực Ứng Dụng:
    - **Nhận dạng chữ viết tay** trong tài liệu số hóa.
    - **Huấn luyện mô hình học sâu** để phân loại hình ảnh.
    - **Thử nghiệm các phương pháp tiền xử lý dữ liệu** để cải thiện độ chính xác của mô hình.
    
    MNIST vẫn là một trong những bộ dữ liệu nền tảng giúp các nhà nghiên cứu và kỹ sư học máy thử nghiệm các thuật toán và mô hình mới.
    """)

def train_model():
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")
    
    mlflow_input()

    # Khởi tạo session state nếu chưa có
    if "run_name" not in st.session_state:
        st.session_state["run_name"] = "default_run"
    if "mlflow_url" not in st.session_state:
        st.session_state["mlflow_url"] = ""

    # Nhập tên thí nghiệm
    st.session_state["run_name"] = st.text_input("🔖 Đặt tên thí nghiệm:", value=st.session_state["run_name"])

    # Load dữ liệu
    Xmt = np.load("X.npy")
    ymt = np.load("y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1) 
    y = ymt.reshape(-1) 

    # Tùy chọn thuật toán
    method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
    n_components = st.slider("Chọn số chiều giảm xuống", 2, 50, 2)

    # Chọn cách trực quan hóa
    visualization_dim = st.radio("Chọn cách trực quan hóa", ["2D", "3D"])
    
    # Nếu chọn t-SNE, thêm tùy chọn Perplexity
    perplexity = 30
    if method == "t-SNE":
        perplexity = st.slider("Chọn Perplexity", 5, 50, 30, step=5)

    # Thanh trượt chọn số lượng mẫu sử dụng từ MNIST
    num_samples = st.slider("Chọn số lượng mẫu MNIST sử dụng:", 1000, 60000, 5000, step=1000)

    # Giới hạn số mẫu để tăng tốc
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    if st.button("🚀 Chạy giảm chiều"):
        with st.spinner("Đang xử lý..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("experiment_name", st.session_state["run_name"])
            mlflow.log_param("method", method)
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("original_dim", X.shape[1])

            if method == "t-SNE":
                mlflow.log_param("perplexity", perplexity)
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            else:
                reducer = PCA(n_components=n_components)

            start_time = time.time()

            for i in range(1, 101):
                time.sleep(0.02)
                progress_bar.progress(i)
                status_text.text(f"🔄 Tiến độ: {i}%")

                if i == 50:
                    X_reduced = reducer.fit_transform(X_subset)

            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)

            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio", explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

            # Hiển thị kết quả
            if visualization_dim == "2D" and n_components >= 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                                 title=f"{method} giảm chiều xuống {n_components}D")
                st.plotly_chart(fig)
            elif visualization_dim == "3D" and n_components >= 3:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                     color=y_subset.astype(str),
                                     title=f"{method} giảm chiều xuống {n_components}D")
                st.plotly_chart(fig)
            else:
                st.warning(f"Không thể hiển thị trực quan với {visualization_dim} khi số chiều = {n_components}!")

            # Lưu kết quả vào MLflow
            os.makedirs("logs", exist_ok=True)
            np.save(f"logs/{method}_X_reduced.npy", X_reduced)
            mlflow.log_artifact(f"logs/{method}_X_reduced.npy")

            mlflow.end_run()
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")

            if st.session_state["mlflow_url"]:
                st.markdown(f"### 🔗 [Truy cập MLflow]({st.session_state['mlflow_url']})")
            else:
                st.warning("⚠️ Chưa có đường link MLflow!")

            progress_bar.empty()
            status_text.empty()


def mlflow_input():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")

    DAGSHUB_USERNAME = "NguyenNhat248"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "Mnist"
    DAGSHUB_TOKEN = "4dd0f9a2823d65298c4840f778a4090d794b30d5"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm MLflow
    mlflow.set_experiment("PCA & t-SNE")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"



def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow với thanh trạng thái tiến trình."""
    st.title("📊 Xem Thí Nghiệm MLflow")

    # Lấy danh sách thí nghiệm
    experiment_name = "PCA & t-SNE"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Không tìm thấy thí nghiệm '{experiment_name}'!")
        return

    st.subheader(f"📌 Thí nghiệm: {experiment_name}")
    st.write(f"**Mã ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Đang hoạt động' if selected_experiment.lifecycle_stage == 'active' else 'Đã xóa'}")
    st.write(f"**Lưu trữ tại:** {selected_experiment.artifact_location}")

    # Lấy danh sách Runs
    st.write("### 🔄 Đang tải danh sách Runs...")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có dữ liệu nào trong thí nghiệm này.")
        return

    total_runs = len(runs)
    run_info = []
    
    progress_bar = st.progress(0)  # Thanh tiến trình

    for i, (_, run) in enumerate(runs.iterrows()):
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_tags = run_data.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")
        created_time = format_time_relative(run_data.info.start_time)
        duration = (run_data.info.end_time - run_data.info.start_time) / 1000 if run_data.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Không rõ")

        run_info.append({
            "Tên Run": run_name,
            "Run ID": run_id,
            "Thời gian tạo": created_time,
            "Thời gian chạy (s)": duration if isinstance(duration, str) else f"{duration:.1f}s",
            "Nguồn": source
        })

        # Cập nhật thanh tiến trình
        progress_bar.progress(int((i + 1) / total_runs * 100))

    progress_bar.empty()  # Xóa thanh tiến trình khi hoàn thành

    # Hiển thị bảng danh sách Runs
    run_info_df = pd.DataFrame(run_info).sort_values(by="Thời gian tạo", ascending=False)
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_info_df, use_container_width=True)

    # Chọn Run từ danh sách
    run_names = run_info_df["Tên Run"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)

    # Lấy Run ID tương ứng
    selected_run_id = run_info_df.loc[run_info_df["Tên Run"] == selected_run_name, "Run ID"].values[0]
    selected_run = mlflow.get_run(selected_run_id)

    # Đổi tên Run
    st.write("### ✏️ Chỉnh sửa tên Run")
    new_run_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_run_name)
            st.success(f"✅ Đã đổi tên thành **{new_run_name}**. Hãy làm mới trang để thấy thay đổi!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")

    # Xóa Run
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Đã xóa run **{selected_run_name}**! Hãy làm mới trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")

    # Hiển thị thông tin chi tiết của Run
    if selected_run:
        st.subheader(f"📌 Chi tiết Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        start_time_ms = selected_run.info.start_time
        start_time = datetime.datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Bắt đầu chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Thông số:")
            st.json(params)

        if metrics:
            st.write("### 📊 Kết quả đo lường:")
            st.json(metrics)

        # Hiển thị model artifact nếu có
        model_artifact_path = f"{st.session_state['mlflow_url']}/{selected_experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Lưu trữ mô hình:")
        st.write(f"📥 [Tải mô hình]({model_artifact_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho Run này.")
def mnist_dim_reduction():
    # Thiết lập CSS để cải thiện giao diện tabs
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            display: flex;
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: thin;
            scrollbar-color: #888 #e0e0e0;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 6px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 3px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background: #e0e0e0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #e0e0e0;
            transition: background-color 0.3s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    ) 

    st.title("🖊️ Ứng dụng PCA & t-SNE trên MNIST")

    # Tạo các tab để dễ dàng điều hướng
    tab_pca, tab_tsne, tab_data, tab_reduction, tab_results = st.tabs([
        "🔍 Giới thiệu PCA", 
        "🔍 Giới thiệu T-SNE", 
        "📊 Xem dữ liệu",  
        "🔀 Giảm số chiều",
        "📈 Kết quả phân tích"
    ])

    with tab_pca: 
        ly_thuyet_PCA() 

    with tab_tsne:
        ly_thuyet_tSne()

    with tab_data: 
        data()    

    with tab_reduction:
        train_model()

    with tab_results: 
        display_mlflow_experiments()    


def run_app(): 
    mnist_dim_reduction()        

if __name__ == "__main__":
    run_app()