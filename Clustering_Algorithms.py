import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import time
import shutil
import humanize
from datetime import datetime
from scipy.stats import mode
from scipy import stats
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Tải dữ liệu MNIST từ OpenML


import streamlit as st

def ly_thuyet_kmeans():
    st.header("📖 Lý thuyết về K-Means")
    
    st.subheader("1️⃣ K-Means là gì?")
    st.write("K-means là một thuật toán **học không giám sát** dùng để phân cụm dữ liệu thành k cụm dựa trên khoảng cách Euclid.")
    
    st.subheader("🎯 Mục tiêu của thuật toán K-Means")
    st.write("Thuật toán **K-Means** tìm các cụm tối ưu trong tập dữ liệu bằng cách tối thiểu hóa tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm của chúng.")
    
    st.subheader("Hàm mục tiêu (Objective Function)")
    st.latex(r"""
    J = \sum_{k=1}^{K} \sum_{x_i \in C_k} || x_i - \mu_k ||^2
    """)
    
    st.write("Trong đó:")
    st.write("- \( K \): Số lượng cụm.")
    st.write("- \( C_k \): Tập hợp các điểm dữ liệu thuộc cụm thứ \( k \).")
    st.write("- \( x_i \): Điểm dữ liệu trong cụm \( C_k \).")
    st.write("- \( \mu_k \): Tâm cụm của \( C_k \).")
    st.write("- \( || x_i - \mu_k ||^2 \): Khoảng cách Euclidean bình phương giữa điểm \( x_i \) và tâm cụm \( \mu_k \).")
    
    st.subheader("2️⃣ Ý tưởng")
    st.write("- Chia tập dữ liệu thành \( K \) cụm, với mỗi cụm có một tâm cụm.")
    st.write("- Dữ liệu được gán vào cụm có tâm cụm gần nhất.")
    st.write("- Cập nhật tâm cụm bằng cách tính trung bình các điểm thuộc cụm.")
    st.write("- Lặp lại cho đến khi không có sự thay đổi đáng kể trong cụm.")
    
    st.subheader("3️⃣ Thuật toán K-Means")
    st.write("1. Chọn số cụm \( K \) (được xác định trước).")
    st.write("2. Khởi tạo \( K \) tâm cụm (chọn ngẫu nhiên hoặc theo K-Means++).")
    st.write("3. Gán dữ liệu vào cụm: Mỗi điểm dữ liệu được gán vào cụm có tâm cụm gần nhất.")
    st.latex(r"""d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}""")
    st.write("4. Cập nhật tâm cụm: Tính lại tâm cụm bằng cách lấy trung bình các điểm trong mỗi cụm.")
    st.latex(r"""\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i""")
    st.write("5. Lặp lại bước 3 & 4 cho đến khi các tâm cụm không thay đổi nhiều nữa hoặc đạt đến số lần lặp tối đa.")
    
    st.subheader("4️⃣ Đánh giá thuật toán K-Means")
    
    st.write("**📌 Elbow Method**")
    st.write("- Tính tổng khoảng cách nội cụm WCSS (Within-Cluster Sum of Squares) cho các giá trị k khác nhau.")
    st.write("- Điểm \"khuỷu tay\" (elbow point) là giá trị k tối ưu, tại đó việc tăng thêm cụm không làm giảm đáng kể WCSS.")
    st.latex(r"""
    WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
    """)
    
    st.write("**📌 Silhouette Score**")
    st.write("- So sánh mức độ gần gũi giữa các điểm trong cụm với các điểm ở cụm khác.")
    st.latex(r"""
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    """)
    st.write("Trong đó:")
    st.write("- \( a(i) \): Khoảng cách trung bình từ điểm \( i \) đến các điểm trong cùng cụm.")
    st.write("- \( b(i) \): Khoảng cách trung bình từ điểm \( i \) đến các điểm trong cụm gần nhất.")
    
    st.write("**📌 Gap Statistic**")
    st.write("- So sánh hiệu quả phân cụm trên dữ liệu thực với dữ liệu ngẫu nhiên (không có cấu trúc).")
    st.latex(r"""
    Gap(k) = \mathbb{E}[\log(W_k^{random})] - \log(W_k^{data})
    """)
    st.write("Trong đó:")
    st.write("- \( W_k^{random} \): WCSS trên random data.")
    st.write("- \( W_k^{data} \): WCSS trên actual data.")

def ly_thuyet_dbscans():
    st.header("📖 Tìm hiểu về DBSCAN")
    
    st.markdown("### 1️⃣ DBSCAN là gì?")
    st.write("DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm dựa trên mật độ. Nó đặc biệt hữu ích khi làm việc với các cụm có hình dạng không xác định và có khả năng phát hiện điểm nhiễu (outlier).")
    
    st.markdown("#### 🔹 Đặc điểm chính của DBSCAN")
    st.write("- **Không cần xác định trước số cụm** như K-Means.")
    st.write("- **Xử lý tốt dữ liệu có nhiễu** bằng cách đánh dấu các điểm không thuộc cụm nào.")
    st.write("- **Có thể phát hiện cụm với hình dạng bất kỳ**, không bị giới hạn vào các hình cầu như K-Means.")
    
    st.markdown("### 2️⃣ Cách hoạt động của DBSCAN")
    
    st.markdown("#### 🔹 Các tham số quan trọng")
    st.write("- **Epsilon (ε):** Xác định bán kính để kiểm tra lân cận của một điểm.")
    st.write("- **MinPts:** Số lượng điểm tối thiểu trong bán kính ε để một điểm được coi là điểm lõi.")
    
    st.markdown("#### 🔹 Phân loại điểm trong DBSCAN")
    st.write("- **Core Point (Điểm lõi):** Có ít nhất MinPts điểm nằm trong vùng ε.")
    st.write("- **Border Point (Điểm biên):** Nằm trong vùng ε của một điểm lõi nhưng không có đủ MinPts để trở thành điểm lõi.")
    st.write("- **Noise (Điểm nhiễu):** Không thuộc vào bất kỳ cụm nào.")
    
    st.markdown("### 3️⃣ Thuật toán DBSCAN")
    st.write("1. Chọn một điểm chưa được kiểm tra.")
    st.write("2. Kiểm tra xem điểm đó có ít nhất MinPts điểm lân cận trong bán kính ε không:")
    st.write("   - Nếu có: Điểm này trở thành **core point**, bắt đầu một cụm mới.")
    st.write("   - Nếu không: Điểm này là **noise**. Tuy nhiên, nếu nó thuộc vùng lân cận của một điểm lõi khác, nó có thể trở thành **border point**.")
    st.write("3. Nếu tìm thấy core point, mở rộng cụm bằng cách thêm tất cả các điểm lân cận vào cụm.")
    st.write("4. Tiếp tục quá trình với các điểm chưa được kiểm tra cho đến khi tất cả các điểm được xử lý.")
    
    st.markdown("### 4️⃣ Đánh giá thuật toán DBSCAN")
    
    st.markdown("#### 🔹 Ưu điểm")
    st.write("- Không yêu cầu số lượng cụm trước.")
    st.write("- Nhận diện được cụm có hình dạng bất kỳ.")
    st.write("- Tốt trong việc xử lý dữ liệu có nhiễu.")
    
    st.markdown("#### 🔹 Nhược điểm")
    st.write("- Lựa chọn tham số ε và MinPts có thể khó khăn.")
    st.write("- Hiệu suất kém với dữ liệu có mật độ không đồng đều.")




def data(): 
    st.title("📚 Giới Thiệu Tập Dữ Liệu MNIST")

    st.markdown("""
    Tập dữ liệu **MNIST (Modified National Institute of Standards and Technology)** là một trong những bộ dữ liệu quan trọng nhất trong lĩnh vực học máy, đặc biệt là trong bài toán nhận dạng chữ số viết tay. Bộ dữ liệu này thường được sử dụng để đánh giá hiệu suất của các mô hình phân loại hình ảnh.

    ## 1️⃣ Tổng Quan Về Tập Dữ Liệu MNIST
    MNIST bao gồm hình ảnh các chữ số từ **0 đến 9**, được viết tay bởi nhiều người khác nhau. Các hình ảnh đã được chuẩn hóa về kích thước và độ sáng, giúp thuận tiện hơn trong quá trình xử lý dữ liệu.  

    Bộ dữ liệu được chia thành hai phần chính:
    - **Tập huấn luyện:** 60.000 hình ảnh.
    - **Tập kiểm tra:** 10.000 hình ảnh.  
      
    Mỗi hình ảnh có độ phân giải **28x28 pixel**, với giá trị cường độ sáng từ **0 đến 255** (0 là màu đen, 255 là màu trắng).

    ## 2️⃣ Mục Đích Sử Dụng MNIST
    Bộ dữ liệu này thường được sử dụng để:  
    - **Huấn luyện mô hình phân loại chữ số viết tay**.
    - **Đánh giá hiệu suất thuật toán học máy**, từ phương pháp truyền thống như KNN, SVM đến mạng nơ-ron nhân tạo (ANN) và mạng nơ-ron tích chập (CNN).
    - **Làm quen với xử lý ảnh số** và các kỹ thuật tiền xử lý như chuẩn hóa dữ liệu, trích xuất đặc trưng.

    ## 3️⃣ Cấu Trúc Dữ Liệu
    Mỗi hình ảnh trong MNIST có thể được biểu diễn dưới dạng ma trận **28x28**, tương ứng với **784 giá trị số**. Khi làm việc với dữ liệu này, có thể:
    - **Chuyển đổi ma trận thành vector 1 chiều** để đưa vào mô hình học máy.
    - **Áp dụng kỹ thuật giảm chiều dữ liệu** như PCA để tối ưu hóa hiệu suất mô hình.
    
    ## 4️⃣ Ứng Dụng Của MNIST
    MNIST được sử dụng rộng rãi trong:
    - **Nhận dạng chữ viết tay**, ứng dụng trong đọc số từ ảnh chụp hoặc tài liệu số hóa.
    - **Nghiên cứu về mạng nơ-ron**, giúp kiểm thử các kiến trúc mới như CNN, GANs.
    - **Học tập và thực hành AI**, là bài toán khởi đầu quen thuộc cho những ai mới tiếp cận học sâu.

    """)



def up_load_db():
    st.header("📥 Tải Dữ Liệu")

    # Kiểm tra nếu dữ liệu đã được tải trước đó
    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔹 **Dữ liệu đã được tải lên!** Bạn có thể tiếp tục với bước tiền xử lý.")
    else:
        option = st.radio("📌 Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Tải lên từ thiết bị"], key="data_source_radio")

        if "data" not in st.session_state:
            st.session_state.data = None

        # Trường hợp tải dữ liệu từ OpenML
        if option == "Tải từ OpenML":
            st.subheader("📂 Tải dữ liệu MNIST từ OpenML")
            if st.button("📥 Bắt đầu tải dữ liệu MNIST", key="download_mnist_button"):
                with st.status("🔄 Đang tải dữ liệu MNIST...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 20):
                        time.sleep(0.5)
                        progress_bar.progress(percent)
                        status.update(label=f"🔄 Đang tải... ({percent}%)")

                    # Tải dữ liệu từ file đã lưu sẵn
                    X = np.load("X.npy")
                    y = np.load("y.npy")

                    status.update(label="✅ Tải dữ liệu thành công!", state="complete")
                    st.session_state.data = (X, y)

        # Trường hợp người dùng muốn tải lên dữ liệu của họ
        else:
            st.subheader("📤 Tải lên dữ liệu của bạn")
            uploaded_file = st.file_uploader("📌 Chọn một file ảnh (PNG, JPG, JPEG)", 
                                             type=["png", "jpg", "jpeg"], 
                                             key="file_upload")

            if uploaded_file is not None:
                with st.status("🔄 Đang xử lý ảnh...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 25):
                        time.sleep(0.3)
                        progress_bar.progress(percent)
                        status.update(label=f"🔄 Đang xử lý... ({percent}%)")

                    image = Image.open(uploaded_file)
                    st.image(image, caption="📷 Ảnh đã tải lên", use_column_width=True)

                    # Kiểm tra kích thước ảnh
                    if image.size != (28, 28):
                        status.update(label="❌ Ảnh không đúng kích thước 28x28 pixel.", state="error")
                    else:
                        status.update(label="✅ Ảnh hợp lệ!", state="complete")
                        image = image.convert('L')  # Chuyển sang ảnh xám
                        image_array = np.array(image).reshape(1, -1)
                        st.session_state.data = image_array

    # Nếu dữ liệu đã sẵn sàng, tiến hành tiền xử lý
    if st.session_state.data is not None:
        st.subheader("✅ Dữ liệu đã sẵn sàng!")

        if isinstance(st.session_state.data, tuple):
            X, y = st.session_state.data
            st.subheader("🔄 Tiền xử lý dữ liệu MNIST")
            preprocess_option = st.selectbox("📌 Chọn phương pháp tiền xử lý:", 
                                            ["Chuẩn hóa (Standardization)", "Giảm chiều (PCA)", "Giữ nguyên"], 
                                            key="preprocess_mnist")

            if preprocess_option == "Chuẩn hóa (Standardization)":
                X_reshaped = X.reshape(X.shape[0], -1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_reshaped)
                st.write("📊 **Dữ liệu sau khi chuẩn hóa:**")
                st.write(pd.DataFrame(X_scaled).head())

            elif preprocess_option == "Giảm chiều (PCA)":
                pca = PCA(n_components=50)
                X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))
                st.write("📊 **Dữ liệu sau khi giảm chiều (PCA):**")
                st.write(pd.DataFrame(X_pca).head())

            else:
                st.write("📊 **Dữ liệu giữ nguyên, không có tiền xử lý.**")

        elif isinstance(st.session_state.data, np.ndarray):
            st.subheader("👁️ Tiền xử lý ảnh")
            preprocess_option_image = st.selectbox("📌 Chọn phương pháp tiền xử lý ảnh:",
                                                   ["Chuẩn hóa ảnh", "Giữ nguyên"], 
                                                   key="preprocess_image")

            if preprocess_option_image == "Chuẩn hóa ảnh":
                image_scaled = st.session_state.data / 255.0
                st.write("📊 **Ảnh sau khi chuẩn hóa:**")
                st.image(image_scaled.reshape(28, 28), caption="Ảnh sau khi chuẩn hóa", use_column_width=True)
            else:
                st.write("📊 **Ảnh giữ nguyên, không có tiền xử lý.**")
    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục.")

    st.markdown("""
    🔹 **Lưu ý:**
    - Dữ liệu ảnh phải có kích thước **28x28 pixel (grayscale)**.
    - Nếu tải từ OpenML, dữ liệu cần có cột **'label'** (số từ 0 đến 9).
    - Nếu dữ liệu không đúng định dạng, hãy sử dụng tập dữ liệu MNIST có sẵn.
    """)



import os
import time
import numpy as np
import streamlit as st
import mlflow
from sklearn.model_selection import train_test_split

def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

    # Tải dữ liệu MNIST
    Xmt = np.load("X.npy")
    ymt = np.load("y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1)  # Giữ nguyên định dạng dữ liệu
    y = ymt.reshape(-1)  

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh dùng để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 
                            min_value=1000, max_value=total_samples, value=10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn tỷ lệ test:", 
                          min_value=0.1, max_value=0.5, value=0.2)

    if st.button("✅ Xác nhận & Lưu"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Cập nhật tiến trình từng bước
        progress_stages = [
            (10, "🔄 Đang chọn số lượng ảnh..."),
            (50, "🔄 Đang chia dữ liệu Train/Test..."),
            (80, "🔄 Đang lưu dữ liệu vào session..."),
            (100, "✅ Hoàn tất!")
        ]

        for progress, message in progress_stages:
            progress_bar.progress(progress)
            status_text.text(f"{message} ({progress}%)")
            time.sleep(0.5)  # Tạo độ trễ để hiển thị tiến trình rõ ràng hơn

        # Chia dữ liệu
        X_selected, y_selected = X[:num_samples], y[:num_samples]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, 
                                                            test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"✅ **Dữ liệu đã được chia:** Train ({len(X_train)}), Test ({len(X_test)})")

    if "X_train" in st.session_state:
        st.write("📌 **Dữ liệu train/test đã sẵn sàng để sử dụng!**")

# 🛠️ Thiết lập MLflow Tracking với DAGsHub (ẩn Access Token)
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
    mlflow.set_experiment("Clustering Algorithms")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


def train():
    st.header("⚙️ Lựa chọn thuật toán & Bắt đầu huấn luyện")

    # Kiểm tra dữ liệu trước khi huấn luyện
    if "X_train" not in st.session_state:
        st.warning("⚠️ Vui lòng thực hiện bước chia dữ liệu trước khi tiếp tục!")
        return

    # Trích xuất dữ liệu từ session_state
    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]
    X_train_prepared = (X_train / 255.0).reshape(X_train.shape[0], -1)  # Chuẩn hóa dữ liệu

    # Chọn mô hình cần sử dụng
    model_option = st.selectbox("🔎 Lựa chọn mô hình:", ["K-Means", "DBSCAN"])
    run_label = st.text_input("📌 Đặt tên cho phiên chạy:", "Run_ML").strip()

    if model_option == "K-Means":
        st.subheader("🔹 Mô hình K-Means Clustering")
        cluster_count = st.slider("📊 Số cụm (K):", min_value=2, max_value=20, value=10)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_prepared)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)

    elif model_option == "DBSCAN":
        st.subheader("🛠️ Mô hình DBSCAN Clustering")
        epsilon = st.slider("📏 Giá trị eps (Bán kính lân cận):", min_value=0.1, max_value=10.0, value=0.5)
        min_samples = st.slider("🔢 Số điểm tối thiểu để tạo cụm:", min_value=2, max_value=20, value=5)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_prepared)
        model = DBSCAN(eps=epsilon, min_samples=min_samples)

    mlflow_input()  # Kết nối với MLflow

    if st.button("🚀 Bắt đầu huấn luyện"):
        progress = st.progress(0)
        status_message = st.empty()

        with mlflow.start_run(run_name=run_label):
            for step in range(0, 101, 10):
                time.sleep(0.5)
                progress.progress(step)
                status_message.text(f"🔄 Đang huấn luyện... {step}% hoàn thành")

            model.fit(X_train_pca)
            progress.progress(100)
            status_message.text("✅ Hoàn thành quá trình huấn luyện!")

            labels = model.labels_

            if model_option == "K-Means":
                label_dict = {}
                for i in range(cluster_count):
                    mask = labels == i
                    if np.sum(mask) > 0:
                        common_label = stats.mode(y_train[mask], keepdims=True).mode[0]
                        label_dict[i] = common_label

                predicted_labels = np.array([label_dict[label] for label in labels])
                train_accuracy = np.mean(predicted_labels == y_train)
                st.write(f"🎯 **Độ chính xác trên tập huấn luyện:** `{train_accuracy * 100:.2f}%`")

                # Lưu thông tin vào MLflow
                mlflow.log_param("model", "K-Means")
                mlflow.log_param("clusters", cluster_count)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.sklearn.log_model(model, "kmeans_model")

            elif model_option == "DBSCAN":
                total_clusters = len(set(labels) - {-1})
                noise_percentage = np.sum(labels == -1) / len(labels)
                st.write(f"🔍 **Số cụm tìm thấy:** `{total_clusters}`")
                st.write(f"🚨 **Tỉ lệ điểm nhiễu:** `{noise_percentage * 100:.2f}%`")

                # Lưu thông tin vào MLflow
                mlflow.log_param("model", "DBSCAN")
                mlflow.log_param("eps", epsilon)
                mlflow.log_param("min_samples", min_samples)
                mlflow.log_metric("total_clusters", total_clusters)
                mlflow.log_metric("noise_ratio", noise_percentage)
                mlflow.sklearn.log_model(model, "dbscan_model")

            if "models" not in st.session_state:
                st.session_state["models"] = []

            # Lưu lại mô hình vào session_state
            st.session_state["models"].append({"name": run_label, "model": model})
            st.success(f"✅ Mô hình `{run_label}` đã được lưu thành công!")

            # Hiển thị danh sách các mô hình đã train
            st.write("📋 **Danh sách các mô hình đã huấn luyện:**")
            for m in st.session_state["models"]:
                st.write(f"🔹 {m['name']}")

            mlflow.end_run()
            st.markdown(f"🔗 [Truy cập MLflow để xem kết quả]({st.session_state['mlflow_url']})")


def du_doan():
    st.header("🔍 Dự đoán Cụm từ Ảnh hoặc CSV")

    # Kiểm tra sự tồn tại của mô hình và nhãn cụm
    if "cluster_model" in st.session_state and "cluster_labels" in st.session_state:
        uploaded_file = st.file_uploader("📤 Tải lên ảnh (28x28, grayscale) hoặc file CSV", type=["png", "jpg", "csv"])
        actual_label = st.text_input("✍️ Nhập nhãn thực tế (nếu có):")

        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                # Xử lý tệp CSV
                data = pd.read_csv(uploaded_file)
                img_vector = data.iloc[0].values.flatten() / 255.0  # Chuẩn hóa dữ liệu
            else:
                # Xử lý tệp ảnh
                image = Image.open(uploaded_file).convert("L").resize((28, 28))
                img_vector = np.array(image).flatten() / 255.0  # Chuyển đổi thành vector 1D

            if st.button("🚀 Tiến hành dự đoán"):
                model = st.session_state["cluster_model"]

                if isinstance(model, KMeans):
                    cluster_id = model.predict([img_vector])[0]
                elif isinstance(model, DBSCAN):
                    # Dự đoán cụm bằng cách tìm điểm gần nhất trong dữ liệu đã phân cụm
                    distances = np.linalg.norm(model.components_ - img_vector, axis=1)
                    cluster_id = model.labels_[np.argmin(distances)]

                # Lấy thông tin nhãn cụm từ session_state
                cluster_labels = st.session_state["cluster_labels"]
     


def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Giao diện quản lý MLflow Experiments."""
    st.title("📊 Theo dõi MLflow Experiments")

    mlflow_input()

    # Xác định experiment cần hiển thị
    experiment_name = "Clustering Algorithms"
    experiments = mlflow.search_experiments()
    experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not experiment:
        st.error(f"🚫 Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"🆔 **Experiment ID:** `{experiment.experiment_id}`")
    st.write(f"📌 **Trạng thái:** {'Hoạt động' if experiment.lifecycle_stage == 'active' else 'Đã xóa'}")
    st.write(f"📂 **Lưu trữ tại:** `{experiment.artifact_location}`")

    # Lấy danh sách Runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Chưa có run nào trong experiment này.")
        return

    # Xử lý thông tin runs
    run_data = []
    for _, run in runs.iterrows():
        run_info = mlflow.get_run(run["run_id"])
        run_name = run_info.data.tags.get("mlflow.runName", f"Run {run['run_id'][:8]}")
        start_time = format_time_relative(run_info.info.start_time)
        duration = ((run_info.info.end_time - run_info.info.start_time) / 1000) if run_info.info.end_time else "Đang chạy"
        source = run_info.data.tags.get("mlflow.source.name", "Không rõ")

        run_data.append({
            "Tên Run": run_name,
            "Run ID": run["run_id"],
            "Bắt đầu": start_time,
            "Thời gian (s)": f"{duration:.1f}s" if isinstance(duration, float) else duration,
            "Nguồn": source
        })

    # Sắp xếp runs theo thời gian gần nhất
    run_df = pd.DataFrame(run_data).sort_values(by="Bắt đầu", ascending=False)

    # Hiển thị danh sách Runs
    st.write("### 🏃‍♂️ Danh sách Runs")
    st.dataframe(run_df, use_container_width=True)

    # Chọn một Run để xem chi tiết
    selected_run_name = st.selectbox("🔍 Chọn Run:", run_df["Tên Run"])
    selected_run_id = run_df.loc[run_df["Tên Run"] == selected_run_name, "Run ID"].values[0]
    selected_run = mlflow.get_run(selected_run_id)

    # --- 📝 ĐỔI TÊN RUN ---
    st.write("### ✏️ Đổi tên Run")
    new_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu thay đổi"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_name)
            st.success(f"✅ Đã đổi tên thành **{new_name}**! Hãy tải lại trang để cập nhật.")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")

    # --- 🗑️ XÓA RUN ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Run **{selected_run_name}** đã bị xóa!")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")

    # --- HIỂN THỊ CHI TIẾT RUN ---
    if selected_run:
        st.subheader(f"📌 Chi tiết Run: {selected_run_name}")
        st.write(f"🆔 **Run ID:** `{selected_run_id}`")
        st.write(f"📌 **Trạng thái:** `{selected_run.info.status}`")
        st.write(f"🕒 **Thời gian bắt đầu:** `{format_time_relative(selected_run.info.start_time)}`")

        # Hiển thị Parameters và Metrics
        if selected_run.data.params:
            st.write("### ⚙️ Tham số:")
            st.json(selected_run.data.params)

        if selected_run.data.metrics:
            st.write("### 📊 Metrics:")
            st.json(selected_run.data.metrics)

        # Hiển thị link tải mô hình nếu có
        model_url = f"{st.session_state['mlflow_url']}/{experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Tệp mô hình:")
        st.write(f"📥 [Tải xuống mô hình]({model_url})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho Run này.")



def ClusteringAlgorithms():
    # CSS Tùy chỉnh để thiết kế giao diện đẹp hơn
    st.markdown(
        """
        <style>
        .custom-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }
        .custom-tab {
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            background: #f8f9fa;
            transition: background 0.3s ease, transform 0.2s ease;
        }
        .custom-tab:hover {
            background: #e9ecef;
            transform: scale(1.05);
        }
        .custom-tab-selected {
            background: #007bff;
            color: white;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎨 MNIST Clustering App")

    # Chọn tab bằng radio button thay vì tabs mặc định
    tab_options = [
        "📘 K-MEANS", "📘 DBSCAN", "📊 Dữ liệu", "📥 Tải dữ liệu",
        "🔀 Chia dữ liệu", "🤖 Phân cụm", "🔍 Kết quả"
    ]
    
    # Hiển thị tab lựa chọn với radio button để tạo trải nghiệm khác
    selected_tab = st.radio("🔹 Chọn một mục:", tab_options, horizontal=True)

    # Hiển thị nội dung theo tab được chọn
    st.markdown("<div class='custom-tabs'>", unsafe_allow_html=True)
    
    if selected_tab == "📘 K-MEANS":
        ly_thuyet_kmeans()

    elif selected_tab == "📘 DBSCAN":
        ly_thuyet_dbscans()

    elif selected_tab == "📊 Dữ liệu":
        data()

    elif selected_tab == "📥 Tải dữ liệu":
        up_load_db()

    elif selected_tab == "🔀 Chia dữ liệu":
        chia_du_lieu()

    elif selected_tab == "🤖 Phân cụm":
        train()

    elif selected_tab == "🔍 Kết quả":
        display_mlflow_experiments()

    st.markdown("</div>", unsafe_allow_html=True)


def run():
    ClusteringAlgorithms()

if __name__ == "__main__":
    run()
