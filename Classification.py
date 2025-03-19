import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openml
import joblib
import shutil
import cv2
import pandas as pd
import time
import os
import mlflow
import humanize
from skimage.transform import resize
from datetime import datetime
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from mlflow.tracking import MlflowClient


def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree") 

    st.markdown("### 1️⃣ Decision Tree là gì?")
    st.write("""
    Decision Tree (Cây quyết định) là một thuật toán học có giám sát được sử dụng trong **phân loại (classification)** và **hồi quy (regression)**.
    Nó hoạt động bằng cách chia dữ liệu thành các nhóm nhỏ hơn dựa trên các điều kiện được thiết lập tại các **nút (nodes)** của cây.
    """)

    st.markdown("### 2️⃣ Ý tưởng") 
    st.markdown("""
    **2.1 Vấn đề cần giải quyết:**  
    - Xác định thứ tự thuộc tính để chia dữ liệu.
    - Do có nhiều thuộc tính và giá trị khác nhau, tìm giải pháp tối ưu toàn cục là không khả thi.
    - Giải pháp: **Phương pháp tham lam (greedy)** → Chọn thuộc tính **tốt nhất** tại mỗi bước dựa trên tiêu chí nhất định.
    """)

    st.markdown("""
    **2.2 Quá trình chia nhỏ dữ liệu:**  
    - Dữ liệu được chia thành **child node** dựa trên thuộc tính được chọn.
    - Lặp lại quá trình này cho đến khi đạt điều kiện dừng.
    """)

    st.markdown("""
    **2.3 Hàm số Entropy:**  
    - Entropy đo **độ hỗn loạn (impurity)** của tập dữ liệu.
    - Công thức:
    """)
    st.latex(r"H(p) = - \sum_{i=1}^{n} p_i \log(p_i)")
    
    st.markdown("""
    **Ý nghĩa của Entropy trong phân phối xác suất:**  
    - **Entropy = 0** khi tập dữ liệu chỉ chứa một nhãn duy nhất (độ chắc chắn cao).  
    - **Entropy cao** khi dữ liệu phân bố đồng đều giữa nhiều nhãn (độ không chắc chắn lớn).
    """)

    st.markdown("### 3️⃣ Thuật toán ID3")
    st.markdown("**Tính toán Entropy tại một Node:**")
    st.latex(r"H(S) = - \sum_{c=1}^{C} \frac{N_c}{N} \log \left(\frac{N_c}{N} \right)")

    st.markdown("**Entropy sau khi phân chia theo thuộc tính x:**")
    st.latex(r"H(x,S) = \sum_{k=1}^{K} \frac{m_k}{N} H(S_k)")

    st.markdown("**Information Gain – Tiêu chí chọn thuộc tính:**")
    st.latex(r"G(x,S) = H(S) - H(x,S)")

    st.markdown("ID3 chọn thuộc tính \\( x^* \\) sao cho **Information Gain** lớn nhất:")
    st.latex(r"x^* = \arg\max_{x} G(x,S) = \arg\min_{x} H(x,S)")

    st.markdown("""
    **Khi nào dừng phân chia?**  
    - ✅ Tất cả dữ liệu trong node thuộc cùng một class.  
    - ✅ Không còn thuộc tính nào để chia tiếp.  
    - ✅ Số lượng điểm dữ liệu trong node quá nhỏ.
    """)



def ly_thuyet_SVM():
    st.header("📖 Lý thuyết về SVM")

    st.markdown("### 1️⃣ SVM là gì?")
    st.write("""
    - Support Vector Machine (SVM) là một thuật toán học có giám sát dùng cho **phân loại** và **hồi quy**.
    - Mục tiêu của SVM là tìm ra **siêu phẳng tối ưu** để phân tách dữ liệu với **khoảng cách lề (margin)** lớn nhất.
    """)

    st.markdown("### 2️⃣ Ý tưởng của SVM")

    st.markdown("#### 2.1 Tìm siêu phẳng phân tách tối ưu")
    st.write("""
    Một siêu phẳng (hyperplane) trong không gian đặc trưng có dạng:
    """)
    st.latex(r"w \cdot x + b = 0")

    st.write("""
    Trong đó:
    - $w$ là vector pháp tuyến của siêu phẳng.
    - $x$ là điểm dữ liệu.
    - $b$ là hệ số điều chỉnh độ dịch chuyển của siêu phẳng.

    Mục tiêu của SVM là tìm siêu phẳng có khoảng cách lớn nhất tới các điểm gần nhất thuộc hai lớp khác nhau (các support vectors).
    """)

    st.markdown("#### 2.2 Tối đa hóa lề (Maximum Margin)")
    st.write("""
    Lề (margin) là khoảng cách giữa siêu phẳng và các điểm dữ liệu gần nhất thuộc hai lớp.  
    SVM cố gắng **tối đa hóa lề** để đảm bảo mô hình có khả năng tổng quát hóa tốt nhất.
    """)

    st.latex(r"D = \frac{|w^T x_0 + b|}{||w||_2}")

    st.markdown("""
    **Trong đó:**
    - $w^T x_0$ là tích vô hướng giữa vector pháp tuyến của siêu phẳng và điểm $x_0$.
    - $||w||_2$ là độ dài (norm) của vector pháp tuyến $w$, được tính bằng:
    """)
    st.latex(r"||w||_2 = \sqrt{w_1^2 + w_2^2 + \dots + w_n^2}")

    st.markdown("#### 2.3 Khi dữ liệu không tách được tuyến tính")
    st.write("""
    - Trong trường hợp dữ liệu không thể phân tách tuyến tính, SVM sử dụng **hàm kernel** để ánh xạ dữ liệu sang không gian bậc cao hơn.
    """)

    st.markdown("#### Các kernel phổ biến:")
    st.write("""
    - **Linear Kernel**: Sử dụng khi dữ liệu có thể phân tách tuyến tính.
    - **Polynomial Kernel**: Ánh xạ dữ liệu sang không gian bậc cao hơn.
    - **RBF (Radial Basis Function) Kernel**: Tốt cho dữ liệu phi tuyến tính.
    - **Sigmoid Kernel**: Mô phỏng như mạng neural.
    """)

    st.markdown("#### 2.4 Vị trí tương đối với siêu phẳng")
    st.write("""
    - **Nếu** $w^T x + b > 0$: Điểm $x$ thuộc **lớp dương**.
    - **Nếu** $w^T x + b < 0$: Điểm $x$ thuộc **lớp âm**.
    - **Nếu** $w^T x + b = 0$: Điểm $x$ nằm **trên siêu phẳng phân tách**.
    """)



def data():
    st.title("🔍 Khám Phá Tập Dữ Liệu MNIST")
    
    # Giới thiệu tổng quan
    st.header("📌 Giới thiệu")
    st.write(
        "Tập dữ liệu MNIST (Modified National Institute of Standards and Technology) "
        "là một trong những bộ dữ liệu phổ biến nhất trong lĩnh vực Machine Learning và Nhận dạng hình ảnh. "
        "Nó chứa các chữ số viết tay, thường được sử dụng để thử nghiệm các thuật toán phân loại."
    )
    
    st.image("https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp", use_container_width=True)
    
    # Thông tin chi tiết về dữ liệu
    st.subheader("📂 Thông tin chi tiết")
    st.markdown(
        "- **Tổng số ảnh:** 70.000 ảnh số viết tay (0 - 9)\n"
        "- **Kích thước ảnh:** 28x28 pixel (grayscale)\n"
        "- **Dữ liệu ảnh:** Mỗi ảnh được biểu diễn bởi ma trận 28x28 với giá trị pixel từ 0 đến 255\n"
        "- **Nhãn:** Số nguyên từ 0 đến 9 tương ứng với chữ số thực tế"
    )
    
    # Lịch sử & ứng dụng
    st.header("📜 Nguồn gốc & Ứng dụng")
    st.write(
        "Bộ dữ liệu MNIST được phát triển từ dữ liệu chữ số viết tay gốc của NIST, "
        "và được chuẩn bị bởi Yann LeCun, Corinna Cortes, và Christopher Burges."
    )
    
    st.subheader("📌 Ứng dụng chính")
    st.markdown(
        "- Đánh giá hiệu suất của các mô hình học máy và học sâu.\n"
        "- Kiểm thử thuật toán nhận dạng chữ số viết tay.\n"
        "- Thực hành xử lý ảnh, phân loại, và học máy.\n"
        "- So sánh các phương pháp trích xuất đặc trưng và mô hình học sâu."
    )
    
    # Phân chia dữ liệu
    st.header("📊 Cấu trúc Tập Dữ Liệu")
    st.markdown(
        "- **Tập huấn luyện:** 60.000 ảnh để dạy mô hình.\n"
        "- **Tập kiểm thử:** 10.000 ảnh để đánh giá mô hình.\n"
        "- **Phân bố đồng đều** giữa các chữ số 0-9 để đảm bảo tính khách quan."
    )
    
    # Các phương pháp tiếp cận
    st.header("🛠️ Phương pháp Tiếp Cận")
    st.subheader("📌 Trích xuất đặc trưng")
    st.write("Các phương pháp truyền thống để xử lý ảnh MNIST:")
    st.markdown("- PCA, HOG, SIFT")
    
    st.subheader("📌 Thuật toán Học Máy")
    st.write("Những thuật toán cơ bản có thể áp dụng:")
    st.markdown("- KNN, SVM, Random Forest, Logistic Regression")
    
    st.subheader("📌 Học Sâu")
    st.write("Các kiến trúc mạng nơ-ron phổ biến để xử lý MNIST:")
    st.markdown("- MLP, CNN (LeNet-5, AlexNet, ResNet), RNN")
    
    

def up_load_db():
    st.title("📥 MNIST Data Loader")
    
    if "mnist_data" in st.session_state and st.session_state.mnist_data is not None:
        st.success("✅ Dữ liệu MNIST đã sẵn sàng!")
    else:
        st.subheader("🔄 Tải dữ liệu từ OpenML")
        if st.button("📂 Tải MNIST", key="download_mnist"):
            st.info("⏳ Đang tải dữ liệu... Vui lòng chờ")
            
            progress = st.progress(0)
            status_text = st.empty()
            for i in range(100):
                time.sleep(0.3 / 10)
                progress.progress(i + 1)
                status_text.text(f"⏳ Đang tải... {i + 1}%")
            
            X = np.load("X.npy")
            y = np.load("y.npy")
            
            st.session_state.mnist_data = (X, y)
            st.success("✅ Tải dữ liệu thành công!")
            progress.empty()
            status_text.empty()
    
    if "mnist_data" in st.session_state and st.session_state.mnist_data is not None:
        X, y = st.session_state.mnist_data
        st.subheader("🎨 Xem trước dữ liệu")
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Dự đoán: {y[i]}", fontsize=10)
            axes[i].axis('off')
        st.pyplot(fig)
        
        st.subheader("🛠️ Chọn phương pháp tiền xử lý")
        preprocess = st.radio("Chọn phương pháp:", ["Không xử lý", "Chuẩn hóa", "Tiêu chuẩn hóa", "Xử lý thiếu"], index=0)
        
        X_reshaped = X.reshape(X.shape[0], -1)
        progress = st.progress(0)
        
        for i in range(100):
            time.sleep(0.2 / 10)
            progress.progress(i + 1)
        
        if preprocess == "Chuẩn hóa":
            X_processed = MinMaxScaler().fit_transform(X_reshaped)
            st.success("✅ Dữ liệu đã được chuẩn hóa!")
        elif preprocess == "Tiêu chuẩn hóa":
            X_processed = StandardScaler().fit_transform(X_reshaped)
            st.success("✅ Dữ liệu đã được tiêu chuẩn hóa!")
        elif preprocess == "Xử lý thiếu":
            X_processed = SimpleImputer(strategy='mean').fit_transform(X_reshaped)
            st.success("✅ Dữ liệu thiếu đã được xử lý!")
        else:
            X_processed = X_reshaped
            st.info("🔹 Không thực hiện tiền xử lý")
        
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(X_processed[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Dự đoán: {y[i]}", fontsize=10)
            axes[i].axis('off')
        st.pyplot(fig)
        
        progress.empty()
    else:
        st.warning("⚠️ Hãy tải dữ liệu trước khi tiếp tục!")

    if __name__ == "__main__":
        load_mnist_data

def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

    # Load dữ liệu
    X = np.load("X.npy")
    y = np.load("y.npy")
    total_samples = X.shape[0]

    # Kiểm tra trạng thái session
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Chọn số lượng ảnh train
    num_samples = st.slider("📌 Số lượng mẫu train:", 1000, total_samples, 10000)

    # Chọn tỷ lệ tập test & validation
    test_ratio = st.slider("📌 Tỷ lệ % dữ liệu Test", 10, 50, 20)
    remaining_ratio = 100 - test_ratio
    val_ratio = st.slider("📌 Tỷ lệ % Validation trong tập Train", 0, 50, 15)

    # Hiển thị thông tin phân chia
    train_ratio = remaining_ratio - val_ratio
    st.write(f"📌 **Tỷ lệ phân chia:** Train={train_ratio}%, Validation={val_ratio}%, Test={test_ratio}%")

    # Khi nhấn nút xác nhận
    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True  

        # Chia tập dữ liệu ban đầu
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

        # Chia tập Train/Test
        stratify_opt = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_ratio / 100, stratify=stratify_opt, random_state=42
        )

        # Chia tập Train/Validation
        stratify_opt = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio / (100 - test_ratio),
            stratify=stratify_opt, random_state=42
        )

        # Lưu dữ liệu vào session_state
        st.session_state.update({
            "total_samples": num_samples,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "train_size": X_train.shape[0], "val_size": X_val.shape[0], "test_size": X_test.shape[0]
        })

        # Hiển thị kết quả chia dữ liệu
        st.success("✅ Dữ liệu đã chia thành công!")
        st.table(pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        }))

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")




def train():
    """Huấn luyện mô hình Decision Tree hoặc SVM và lưu trên MLflow với thanh tiến trình hiển thị %."""
    mlflow_input()

    # 📥 Kiểm tra dữ liệu
    if not all(key in st.session_state for key in ["X_train", "y_train", "X_test", "y_test"]):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
    X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]

    # 🌟 Chuẩn hóa dữ liệu
    X_train, X_test = X_train.reshape(-1, 28 * 28) / 255.0, X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 Đặt tên thí nghiệm
    experiment_name = st.text_input("📌 Đặt tên thí nghiệm:", "default_experiment", 
                                    help="Tên của thí nghiệm để dễ dàng quản lý trên MLflow.")

    # 📌 Lựa chọn mô hình
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
    
    if model_choice == "Decision Tree":
        criterion = st.selectbox("Criterion (Hàm mất mát: Gini/Entropy) ", ["gini", "entropy"])
        max_depth = st.slider("max_depth", 1, 20, 5, help="Giới hạn độ sâu của cây để tránh overfitting.")
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    else:
        C = st.slider("C (Hệ số điều chuẩn)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel (Hàm nhân)", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    # 📌 Chọn số folds cho KFold Cross-Validation
    k_folds = st.slider("Số folds", 2, 10, 5, help="Số tập chia để đánh giá mô hình.")

    # 🚀 Bắt đầu huấn luyện
    if st.button("Huấn luyện mô hình"):
        with st.spinner("🔄 Đang huấn luyện mô hình..."):
            progress_bar = st.progress(0)
            percent_text = st.empty()  # Chỗ hiển thị %

            with mlflow.start_run(run_name=experiment_name):
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                cv_scores = []

                # Vòng lặp Cross-Validation
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    val_pred = model.predict(X_val_fold)
                    val_acc = accuracy_score(y_val_fold, val_pred)
                    cv_scores.append(val_acc)
                    mlflow.log_metric("cv_accuracy", val_acc, step=fold)

                    # Cập nhật thanh trạng thái (bỏ qua hiển thị từng fold)
                    percent_done = int(((fold + 1) / k_folds) * 70)
                    progress_bar.progress(percent_done)
                    percent_text.write(f"**Tiến độ: {percent_done}%**")

                    time.sleep(1)  

                # Kết quả CV
                cv_accuracy_mean = np.mean(cv_scores)
                cv_accuracy_std = np.std(cv_scores)
                st.success(f"✅ **Cross-Validation Accuracy:** {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")

                # Huấn luyện trên toàn bộ tập train
                model.fit(X_train, y_train)

                # Cập nhật tiến trình (85%)
                progress_bar.progress(85)
                percent_text.write("**Tiến độ: 85%**")

                # Dự đoán trên test set
                y_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("test_accuracy", test_acc)
                st.success(f"✅ **Độ chính xác trên test set:** {test_acc:.4f}")

                # Delay thêm 20s trước khi hoàn thành
                for i in range(1, 21):
                    progress_percent = 85 + (i // 2)
                    progress_bar.progress(progress_percent)
                    percent_text.write(f"**Tiến độ: {progress_percent}%**")
                    time.sleep(1)

                # Hoàn thành tiến trình
                progress_bar.progress(100)
                percent_text.write("✅ **Tiến độ: 100% - Hoàn thành!**")

                # Log tham số vào MLflow
                mlflow.log_param("experiment_name", experiment_name)
                mlflow.log_param("model", model_choice)
                mlflow.log_param("k_folds", k_folds)
                if model_choice == "Decision Tree":
                    mlflow.log_param("criterion", criterion)
                    mlflow.log_param("max_depth", max_depth)
                else:
                    mlflow.log_param("C", C)
                    mlflow.log_param("kernel", kernel)

                mlflow.log_metric("cv_accuracy_mean", cv_accuracy_mean)
                mlflow.log_metric("cv_accuracy_std", cv_accuracy_std)
                mlflow.sklearn.log_model(model, model_choice.lower())

                st.success(f"✅ Đã log dữ liệu cho **{experiment_name}**!")
                st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")


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
    mlflow.set_experiment("Classifications")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()

def preprocess_canvas_image(canvas_result):
    """Xử lý ảnh từ canvas: chuyển grayscale, resize 8x8 và chuẩn hóa."""
    if canvas_result.image_data is None:
        return None

    # Lấy kênh alpha (để nhận diện nét vẽ)
    img = canvas_result.image_data[:, :, 3] * 255  # Chuyển alpha về 0-255
    img = Image.fromarray(img.astype(np.uint8))  # Chuyển thành ảnh PIL
    
    # Resize về 8x8 (đúng với mô hình SVM digits)
    img = img.resize((8, 8)).convert("L")

    # Chuyển sang numpy array, chuẩn hóa về [0, 16] (giống sklearn digits dataset)
    img = np.array(img, dtype=np.float32)
    img = img / img.max() * 16  # Normalize về [0, 16]

    return img.flatten().reshape(1, -1)  # Chuyển thành vector 1D có 64 features


def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def show_mlflow_experiments():
    """Xem danh sách Runs trong MLflow."""
    st.title("📊 MLflow Experiment Viewer")

    mlflow_input()

    experiment_target = "Classifications"
    available_experiments = mlflow.search_experiments()
    chosen_experiment = next((exp for exp in available_experiments if exp.name == experiment_target), None)

    if not chosen_experiment:
        st.error(f"❌ Không tìm thấy Experiment '{experiment_target}'!")
        return

    st.subheader(f"📌 Experiment: {experiment_target}")
    st.write(f"**Experiment ID:** {chosen_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if chosen_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Lưu trữ tại:** {chosen_experiment.artifact_location}")

    # Lấy danh sách runs từ MLflow
    experiment_runs = mlflow.search_runs(experiment_ids=[chosen_experiment.experiment_id])

    if experiment_runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    # Xử lý danh sách Runs
    runs_list = []
    for _, run_entry in experiment_runs.iterrows():
        run_id = run_entry["run_id"]
        run_details = mlflow.get_run(run_id)
        run_tags = run_details.data.tags
        run_display_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")
        run_creation_time = format_time_relative(run_details.info.start_time)
        run_duration = (run_details.info.end_time - run_details.info.start_time) / 1000 if run_details.info.end_time else "Đang chạy"
        run_origin = run_tags.get("mlflow.source.name", "Unknown")

        runs_list.append({
            "Run Name": run_display_name,
            "Run ID": run_id,
            "Created": run_creation_time,
            "Duration (s)": run_duration if isinstance(run_duration, str) else f"{run_duration:.1f}s",
            "Source": run_origin
        })

    # Sắp xếp runs theo thời gian tạo (gần nhất trước)
    runs_dataframe = pd.DataFrame(runs_list).sort_values(by="Created", ascending=False)

    # Hiển thị bảng danh sách Runs
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(runs_dataframe, use_container_width=True)

    # Chọn một Run để xem chi tiết
    available_run_names = runs_dataframe["Run Name"].tolist()
    chosen_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", available_run_names)

    # Lấy Run ID tương ứng với Run Name
    chosen_run_id = runs_dataframe.loc[runs_dataframe["Run Name"] == chosen_run_name, "Run ID"].values[0]
    chosen_run = mlflow.get_run(chosen_run_id)

    # --- ✏️ ĐỔI TÊN RUN ---
    st.write("### ✏️ Đổi tên Run")
    updated_run_name = st.text_input("Nhập tên mới:", chosen_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(chosen_run_id, "mlflow.runName", updated_run_name)
            st.success(f"✅ Đã cập nhật tên thành **{updated_run_name}**. Vui lòng tải lại trang để thấy thay đổi!")
        except Exception as err:
            st.error(f"❌ Lỗi khi đổi tên: {err}")

    # --- 🗑️ XÓA RUN ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(chosen_run_id)
            st.success(f"✅ Đã xóa Run **{chosen_run_name}**! Vui lòng tải lại trang để cập nhật danh sách.")
        except Exception as err:
            st.error(f"❌ Lỗi khi xóa run: {err}")

    # --- HIỂN THỊ CHI TIẾT RUN ---
    if chosen_run:
        st.subheader(f"📌 Thông tin Run: {chosen_run_name}")
        st.write(f"**Run ID:** {chosen_run_id}")
        st.write(f"**Trạng thái:** {chosen_run.info.status}")

        # Chuyển đổi timestamp về dạng thời gian đọc được
        start_timestamp = chosen_run.info.start_time
        formatted_start_time = datetime.fromtimestamp(start_timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_timestamp else "Không có dữ liệu"

        st.write(f"**Thời gian bắt đầu:** {formatted_start_time}")

        # Hiển thị parameters và metrics
        run_parameters = chosen_run.data.params
        run_metrics = chosen_run.data.metrics

        if run_parameters:
            st.write("### ⚙️ Parameters:")
            st.json(run_parameters)

        if run_metrics:
            st.write("### 📊 Metrics:")
            st.json(run_metrics)

    else:
        st.warning("⚠ Không có thông tin cho Run đã chọn.")


def preprocess_canvas_image(canvas_result):
    """Tiền xử lý ảnh vẽ từ canvas để phù hợp với mô hình."""
    if canvas_result.image_data is None:
        return None

    img = canvas_result.image_data[:, :, :3]  # Lấy 3 kênh màu
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Chuyển sang grayscale
    img = cv2.resize(img, (8, 8))  # Resize về 8x8 pixels
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Làm mịn ảnh để giảm nhiễu
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)  # Đảo ngược màu (trắng nền đen)
    
    img = img / 255.0  # Chuẩn hóa pixel về [0,1]
    img = img.flatten().reshape(1, -1)  # Chuyển thành vector 1D (1, 64)
    
    return img

def du_doan():
    """Giao diện dự đoán số viết tay hoặc tập dữ liệu test."""
    st.title("🔢 Dự đoán chữ số viết tay")
    option = st.radio("Chọn cách nhập dữ liệu:", ["Vẽ số", "Tải lên tập test"])

    if option == "Vẽ số":
        st.subheader("✏️ Vẽ số vào ô bên dưới:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key="canvas"
        )
    else:
        st.subheader("📂 Tải lên tập dữ liệu để dự đoán")
        file = st.file_uploader("Chọn file dữ liệu (.csv hoặc .npy):", type=["csv", "npy"])
        if file:
            data = pd.read_csv(file).values if file.name.endswith(".csv") else np.load(file)
            st.write(f"📊 Tổng số mẫu test: {data.shape[0]}")

    # Load mô hình
    model_path = "svm_mnist_rbf.joblib"
    model = joblib.load(model_path)
    try:
        model = joblib.load(model_path)
        st.success("✅ Mô hình SVM tuyến tính đã sẵn sàng!")

        # Kiểm tra số lượng features mà mô hình yêu cầu
        print("Mô hình yêu cầu số feature:", model.n_features_in_)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy file `{model_path}`. Hãy kiểm tra lại!")
        return

    # Nếu chọn vẽ số, xử lý ảnh từ canvas
    if option == "Vẽ số" and st.button("🔮 Dự đoán"):
        if canvas_result.image_data is not None:
            img = preprocess_canvas_image(canvas_result)

            # Kiểm tra số features
            print("Shape of processed image:", img.shape)
            print("Mô hình SVM yêu cầu số feature:", model.n_features_in_)

            if img.shape[1] != model.n_features_in_:
                st.error("⚠️ Ảnh đầu vào không có đúng số feature! Hãy kiểm tra lại preprocessing.")
                return

            prediction = model.predict(img)
            st.subheader(f"🔢 Kết quả dự đoán: {prediction[0]}")
        else:
            st.error("⚠️ Vui lòng vẽ một số trước khi dự đoán!")

    # Nếu tải lên tập test
    elif option == "Tải lên tập test" and file and st.button("🔮 Dự đoán toàn bộ"):
        if data.shape[1] != model.n_features_in_:
            st.error(f"⚠️ Số lượng features ({data.shape[1]}) không khớp với mô hình ({model.n_features_in_}).")
            return

        preds = model.predict(data)
        probs = model.decision_function(data) if hasattr(model, 'decision_function') else model.predict_proba(data)
        confidences = np.max(probs, axis=1) if probs is not None else ["Không rõ"] * len(preds)

        st.write("📌 Kết quả trên tập dữ liệu test:")
        for i in range(min(10, len(preds))):
            st.write(f"Mẫu {i+1}: {preds[i]} (Độ tin cậy: {confidences[i]:.2f})")

        # Hiển thị 5 ảnh đầu tiên
        fig, axes = plt.subplots(1, min(5, len(data)), figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(data[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"{preds[i]} ({confidences[i]:.2f})")
            ax.axis("off")
        st.pyplot(fig)

        



def Classification():
    # Thiết lập CSS để hỗ trợ hiển thị tabs với hiệu ứng hover và thanh cuộn
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            scrollbar-width: thin;
            scrollbar-color: #888 #f0f0f0;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 6px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 3px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #f0f0f0;
            transition: background-color 0.3s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Tiêu đề ứng dụng
    st.title("🖥️ MNIST Classification App")

    # Tạo các tab trong giao diện Streamlit
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📖 Lý thuyết Decision Tree", 
        "📖 Lý thuyết SVM", 
        "🚀 Data", 
        "📥 Tải dữ liệu", 
        "⚙️ Huấn luyện", 
        "Tracking mlflow",
        "🔮 Dự đoán"
    ])

    # Nội dung của từng tab
    with tab1:
        ly_thuyet_Decision_tree()

    with tab2:
        ly_thuyet_SVM()
    
    with tab3:
        data()

    with tab4:
        up_load_db()
    
    with tab5:      
        chia_du_lieu()
        train()
    
    with tab6:
        show_mlflow_experiments()  # Thay thế dòng cũ


    with tab7:
        du_doan()  # Gọi hàm dự đoán để xử lý khi vào tab Dự đoán

def run(): 
    Classification()

if __name__ == "__main__":
    run()

