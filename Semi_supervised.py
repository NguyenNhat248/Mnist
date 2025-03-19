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
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split 

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
    mlflow.set_experiment("Semi-supervised")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def show_mlflow_experiments():
    """Hiển thị danh sách các Runs trong MLflow với các tùy chọn quản lý."""
    st.title("📊 Trình quản lý MLflow Experiments")
    
    # Nhập thông tin từ người dùng
    mlflow_input()
    
    # Chọn experiment cần hiển thị
    experiment_name = "Neural Network"
    experiments = mlflow.search_experiments()
    current_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)
    
    if not current_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return
    
    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {current_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if current_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Đường dẫn lưu trữ:** {current_experiment.artifact_location}")
    
    # Lấy danh sách các Runs
    runs = mlflow.search_runs(experiment_ids=[current_experiment.experiment_id])
    
    if runs.empty:
        st.warning("⚠ Hiện không có Runs nào trong Experiment này.")
        return
    
    # Xử lý thông tin Runs để hiển thị
    run_data_list = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_metadata = mlflow.get_run(run_id)
        run_tags = run_metadata.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")
        created_time = format_time_relative(run_metadata.info.start_time)
        duration = ((run_metadata.info.end_time - run_metadata.info.start_time) / 1000) if run_metadata.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Không xác định")

        run_data_list.append({
            "Tên Run": run_name,
            "Run ID": run_id,
            "Thời gian tạo": created_time,
            "Thời gian chạy (s)": f"{duration:.1f}s" if isinstance(duration, float) else duration,
            "Nguồn": source
        })
    
    # Hiển thị danh sách Runs
    df_runs = pd.DataFrame(run_data_list).sort_values(by="Thời gian tạo", ascending=False)
    st.write("### 🏃‍♂️ Danh sách các Runs:")
    st.dataframe(df_runs, use_container_width=True)
    
    # Lựa chọn Run để xem chi tiết
    run_names = df_runs["Tên Run"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)
    selected_run_id = df_runs.loc[df_runs["Tên Run"] == selected_run_name, "Run ID"].values[0]
    selected_run = mlflow.get_run(selected_run_id)
    
    # --- Chỉnh sửa thông tin Run ---
    st.write("### ✏️ Đổi tên Run")
    new_run_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_run_name)
            st.success(f"✅ Tên Run đã được cập nhật thành **{new_run_name}**. Hãy tải lại trang để xem thay đổi!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")
    
    # --- Xóa Run ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Run **{selected_run_name}** đã bị xóa! Hãy tải lại trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa Run: {e}")
    
    # --- Hiển thị chi tiết Run ---
    if selected_run:
        st.subheader(f"📌 Chi tiết Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time
        formatted_start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian bắt đầu:** {formatted_start_time}")
        
        # Hiển thị Parameters và Metrics
        params = selected_run.data.params
        metrics = selected_run.data.metrics
        
        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)
        
        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)
        
        # Đường dẫn Model Artifact
        model_artifact_url = f"{st.session_state['mlflow_url']}/{current_experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Model Artifact:")
        st.write(f"📥 [Tải mô hình]({model_artifact_url})")
    else:
        st.warning("⚠ Không tìm thấy thông tin của Run này.")

def tong_quan():
    st.title("Tổng quan về bộ dữ liệu MNIST")

    st.header("1. Giới thiệu")
    st.write(
        "MNIST (Modified National Institute of Standards and Technology) là một bộ dữ liệu phổ biến "
        "trong lĩnh vực thị giác máy tính và học máy, thường được sử dụng để đào tạo và kiểm thử các "
        "thuật toán nhận diện chữ số viết tay."
    )

    st.subheader("Thông tin chính")
    st.write("- Tổng cộng 70.000 hình ảnh grayscale của các chữ số từ 0 đến 9.")
    st.write("- Mỗi hình có kích thước 28x28 pixel.")
    st.write("- Định dạng dữ liệu: Ma trận 28x28 với giá trị pixel từ 0 (màu đen) đến 255 (màu trắng).")
    st.write("- Mỗi ảnh có nhãn tương ứng là một số nguyên từ 0 đến 9.")

    st.header("2. Nguồn gốc và ứng dụng")
    st.write("- Được xây dựng dựa trên bộ dữ liệu chữ số viết tay của NIST, với sự chuẩn bị của LeCun, Cortes và Burges.")
    st.write("- Là thước đo chuẩn cho hiệu suất của các thuật toán xử lý hình ảnh và mạng nơ-ron.")
    st.write("- Được dùng rộng rãi trong nghiên cứu về trí tuệ nhân tạo và thị giác máy tính.")

    st.header("3. Cấu trúc tập dữ liệu")
    st.write("- **Tập huấn luyện:** 60.000 hình ảnh.")
    st.write("- **Tập kiểm thử:** 10.000 hình ảnh.")
    st.write("- Các chữ số từ 0 đến 9 có phân bố tương đối đồng đều.")

    st.header("4. Ứng dụng thực tế")
    st.write("- Đào tạo mô hình phân loại chữ số viết tay.")
    st.write("- So sánh hiệu suất giữa các thuật toán học máy và học sâu.")
    st.write("- Làm quen với các phương pháp tiền xử lý dữ liệu")

def upload_data():
    st.header("📥 Tải dữ liệu vào hệ thống")

    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔸 **Dữ liệu đã có sẵn!** Bạn có thể chuyển sang bước tiếp theo.")
    else:
        data_source = st.radio("Lựa chọn cách nhập dữ liệu:", ["Lấy từ OpenML", "Tải lên từ máy tính"], key="data_source_option")

        if "data" not in st.session_state:
            st.session_state.data = None

        if data_source == "Lấy từ OpenML":
            st.subheader("📂 Nhập dữ liệu MNIST từ OpenML")
            if st.button("Bắt đầu tải xuống", key="download_openml"):
                with st.status("🔄 Đang lấy dữ liệu từ OpenML...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 20):
                        time.sleep(0.5)
                        progress_bar.progress(percent)
                        status.update(label=f"🔄 Đang xử lý... ({percent}%)")

                    # Giả lập quá trình tải dữ liệu
                    X = np.load("X.npy")
                    y = np.load("y.npy")

                    status.update(label="✅ Dữ liệu đã được tải thành công!", state="complete")

                    st.session_state.data = (X, y)

        else:
            st.subheader("📤 Tải lên file của bạn")
            uploaded_file = st.file_uploader("Chọn file ảnh", type=["png", "jpg", "jpeg"], key="upload_file")

            if uploaded_file is not None:
                with st.status("🔄 Đang xử lý tập tin...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 25):
                        time.sleep(0.3)
                        progress_bar.progress(percent)
                        status.update(label=f"🔄 Đang kiểm tra... ({percent}%)")

                    image = Image.open(uploaded_file).convert('L')
                    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

                    if image.size != (28, 28):
                        status.update(label="❌ Kích thước ảnh không phù hợp! Hãy chọn ảnh có kích thước 28x28 pixel.", state="error")
                    else:
                        status.update(label="✅ Hình ảnh hợp lệ!", state="complete")
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                        image_tensor = transform(image).unsqueeze(0)
                        st.session_state.data = image_tensor

    if st.session_state.data is not None:
        st.subheader("✅ Dữ liệu đã sẵn sàng!")
    else:
        st.warning("⚠ Vui lòng nhập dữ liệu trước khi tiếp tục.")

    st.markdown("""
    🔹 **Lưu ý:**  
    - Chỉ hỗ trợ ảnh grayscale với kích thước **28x28 pixel**.  
    - Khi lấy dữ liệu từ OpenML, cần đảm bảo có cột **'label'** chứa nhãn từ 0 đến 9.  
    - Nếu file không đúng định dạng, vui lòng sử dụng dữ liệu MNIST từ OpenML.  
    """)



def split_data():
    st.title("📌 Phân chia dữ liệu Train/Test")

    # Kiểm tra dữ liệu đã tải lên chưa
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Hãy nhập dữ liệu trước khi tiến hành chia tập!")
        return

    # Lấy dữ liệu từ session_state
    X, y = st.session_state.data
    total_samples = X.shape[0]

    # Nếu chưa có biến cờ "data_split_done", thiết lập mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Chọn số lượng dữ liệu sử dụng để train
    num_samples = st.number_input("📌 Chọn số lượng ảnh để huấn luyện:", min_value=1000, max_value=total_samples, value=20000, step=1000)
    
    # Chọn tỷ lệ train/test
    test_ratio = st.slider("📌 Chọn phần trăm dữ liệu dành cho kiểm thử", 10, 50, 20)
    train_ratio = 100 - test_ratio
    st.write(f"📌 **Tỷ lệ phân chia:** Train={train_ratio}%, Test={test_ratio}%")

    # Placeholder cho tiến trình và bảng kết quả
    progress_placeholder = st.empty()
    results_table = st.empty()

    # Xử lý khi nhấn nút xác nhận
    if st.button("✅ Xác nhận & Lưu", key="save_split"):
        progress_placeholder.progress(10)  # Tiến trình bắt đầu
        st.session_state.data_split_done = True  # Đánh dấu đã chia tập dữ liệu
        
        # Lấy số lượng dữ liệu theo yêu cầu
        if num_samples == total_samples:
            X_selected, y_selected = X, y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples, stratify=y, random_state=42
            )

        progress_placeholder.progress(40)  # Cập nhật tiến trình

        # Phân chia tập Train/Test
        stratify_condition = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=test_ratio/100, stratify=stratify_condition, random_state=42
        )

        progress_placeholder.progress(70)  # Tiến trình 70% hoàn thành

        # Lưu dữ liệu vào session_state
        st.session_state.total_samples = num_samples
        st.session_state["neural_X_train"] = X_train
        st.session_state["neural_X_test"] = X_test
        st.session_state["neural_y_train"] = y_train
        st.session_state["neural_y_test"] = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.train_size = X_train.shape[0]

        progress_placeholder.progress(90)  # Gần hoàn tất

        # Hiển thị kết quả phân chia
        st.session_state.summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]
        })

        st.success("✅ Phân chia dữ liệu thành công!")
        results_table.table(st.session_state.summary_df)
        progress_placeholder.progress(100)  # Hoàn tất

    # Nếu dữ liệu đã được chia trước đó
    if st.session_state.data_split_done:
        if "summary_df" in st.session_state:
            results_table.table(st.session_state.summary_df)
        
        st.info("✅ Dữ liệu đã được phân chia. Nhấn nút dưới để thay đổi nếu cần.")

        if st.button("🔄 Chia lại dữ liệu", key="resplit_data"):
            progress_placeholder.progress(10)  # Tiến trình khởi động lại
            results_table.empty()

            # Chọn lại dữ liệu theo số lượng yêu cầu
            if num_samples == total_samples:
                X_selected, y_selected = X, y
            else:
                X_selected, _, y_selected, _ = train_test_split(
                    X, y, train_size=num_samples, stratify=y, random_state=42
                )

            progress_placeholder.progress(40)

            # Chia train/test lần nữa
            stratify_condition = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_selected, test_size=test_ratio/100, stratify=stratify_condition, random_state=42
            )

            progress_placeholder.progress(70)

            # Cập nhật lại session_state
            st.session_state.total_samples = num_samples
            st.session_state["neural_X_train"] = X_train
            st.session_state["neural_X_test"] = X_test
            st.session_state["neural_y_train"] = y_train
            st.session_state["neural_y_test"] = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.train_size = X_train.shape[0]

            progress_placeholder.progress(90)

            # Cập nhật bảng kết quả mới
            st.session_state.summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]
            })
            st.success("✅ Phân chia lại dữ liệu thành công!")
            results_table.table(st.session_state.summary_df)
            progress_placeholder.progress(100)  # Hoàn tất



#Callback phuc vu train2
class ProgressBarCallback:
    def __init__(self, total_epochs, progress_bar, status_text, max_train_progress=80):
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_train_progress = max_train_progress

    def on_epoch_begin(self, epoch):
        progress = (epoch + 1) / self.total_epochs * self.max_train_progress
        self.progress_bar.progress(min(int(progress), self.max_train_progress))
        self.status_text.text(f"🛠️ Đang huấn luyện mô hình... Epoch {epoch + 1}/{self.total_epochs}")

    def on_train_end(self):
        self.progress_bar.progress(self.max_train_progress)
        self.status_text.text("✅ Huấn luyện mô hình hoàn tất, đang chuẩn bị logging...")

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_layers, num_nodes, activation):
        super(NeuralNet, self).__init__()
        layers = [nn.Linear(input_size, num_nodes), self.get_activation(activation)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(self.get_activation(activation))
        
        layers.append(nn.Linear(num_nodes, 10))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_activation(self, activation):
        """ Trả về hàm kích hoạt phù hợp """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Không hỗ trợ activation: {activation}")

def pseudo_labelling():
    st.header("⚙️ Pseudo Labelling với Mạng Nơ-ron (PyTorch)")
    
    # Kiểm tra dữ liệu đã được chia chưa
    if "neural_X_train" not in st.session_state or "neural_X_test" not in st.session_state:
        st.error("⚠️ Dữ liệu chưa sẵn sàng! Vui lòng thực hiện chia dữ liệu trước.")
        return
    
    # Chuyển đổi dữ liệu về tensor và chuẩn hóa
    X_train_full = torch.tensor(st.session_state["neural_X_train"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    y_train_full = torch.tensor(st.session_state["neural_y_train"], dtype=torch.long)
    X_test = torch.tensor(st.session_state["neural_X_test"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    y_test = torch.tensor(st.session_state["neural_y_test"], dtype=torch.long)
    
    # Lấy một phần nhỏ dữ liệu ban đầu (~1% mỗi lớp)
    X_initial, y_initial = [], []
    for label in range(10):
        label_indices = torch.where(y_train_full == label)[0]
        sample_count = max(1, int(0.01 * len(label_indices)))
        chosen_indices = label_indices[torch.randperm(len(label_indices))[:sample_count]]
        X_initial.append(X_train_full[chosen_indices])
        y_initial.append(y_train_full[chosen_indices])
    
    X_initial = torch.cat(X_initial, dim=0)
    y_initial = torch.cat(y_initial, dim=0)
    mask = torch.ones(len(X_train_full), dtype=torch.bool)
    mask[chosen_indices] = False
    X_unlabeled = X_train_full[mask]
    
    # Tùy chọn chế độ lặp
    iteration_mode = st.selectbox("Chế độ lặp:", ["Số vòng lặp giới hạn", "Gán toàn bộ tập train"], key="iteration_mode")
    
    max_iterations = st.slider("Số vòng tối đa", 1, 10, 5) if iteration_mode == "Số vòng lặp giới hạn" else float('inf')
    if iteration_mode == "Gán toàn bộ tập train":
        st.warning("⚠️ Quá trình có thể mất nhiều thời gian do số vòng lặp không giới hạn!")

    # Cấu hình mô hình
    hidden_layers = st.slider("Số lớp ẩn", 1, 5, 2)
    neurons_per_layer = st.slider("Số node mỗi lớp", 32, 256, 128)
    activation_function = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"])
    training_epochs = st.slider("Số epoch mỗi vòng", 1, 50, 10)
    confidence_threshold = st.slider("Ngưỡng gán nhãn", 0.5, 1.0, 0.95, step=0.01)
    learning_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
    
    run_identifier = st.text_input("🔹 Đặt tên Run:", "Pseudo_Default_Run")
    
    if st.button("Bắt đầu Pseudo Labelling"):
        mlflow.start_run(run_name=f"Pseudo_{run_identifier}")
        model = NeuralNet(28 * 28, hidden_layers, neurons_per_layer, activation_function)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X_labeled, y_labeled = X_initial.clone(), y_initial.clone()
        X_unlabeled_remaining = X_unlabeled.clone()
        total_data_count = len(X_train_full)
        
        iteration = 0
        progress_bar = st.progress(0)
        status_message = st.empty()
        
        while iteration < max_iterations:
            st.write(f"### Vòng lặp {iteration + 1}")
            train_dataset = TensorDataset(X_labeled, y_labeled)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(training_epochs):
                status_message.text(f"🚀 Đang huấn luyện - Epoch {epoch + 1}/{training_epochs}...")
                progress_bar.progress(int((epoch + 1) / training_epochs * 50))
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                time.sleep(0.5)  # Giảm tải cho hệ thống
            
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test)
                test_accuracy = (test_predictions.argmax(dim=1) == y_test).float().mean().item()
            
            st.write(f"📊 Độ chính xác trên tập kiểm thử: {test_accuracy:.4f}")
            mlflow.log_metric("pseudo_test_accuracy", test_accuracy, step=iteration)
            
            if len(X_unlabeled_remaining) == 0:
                break
            
            status_message.text("🔍 Đang gán nhãn giả cho dữ liệu chưa được gán...")
            with torch.no_grad():
                unlabeled_outputs = model(X_unlabeled_remaining)
                probabilities, predicted_labels = unlabeled_outputs.softmax(dim=1).max(dim=1)
            
            confident_mask = probabilities >= confidence_threshold
            X_confident = X_unlabeled_remaining[confident_mask]
            y_confident = predicted_labels[confident_mask]
            
            st.write(f"Số mẫu mới được gán nhãn: {X_confident.shape[0]} (Ngưỡng: {confidence_threshold})")
            st.write(f"Còn lại chưa gán nhãn: {X_unlabeled_remaining.shape[0] - X_confident.shape[0]}")
            
            if len(X_confident) == 0:
                break
            
            X_labeled = torch.cat([X_labeled, X_confident])
            y_labeled = torch.cat([y_labeled, y_confident])
            X_unlabeled_remaining = X_unlabeled_remaining[~confident_mask]
            
            labeled_fraction = X_labeled.shape[0] / total_data_count
            progress_bar.progress(min(int(50 + 50 * labeled_fraction), 100))
            status_message.text(f"📈 Tiến trình gán nhãn: {X_labeled.shape[0]}/{total_data_count} mẫu ({labeled_fraction:.2%})")
            
            iteration += 1
            if iteration_mode == "Gán toàn bộ tập train" and len(X_unlabeled_remaining) == 0:
                break
        
        torch.save(model.state_dict(), "pseudo_model_final.pth")
        mlflow.log_artifact("pseudo_model_final.pth")
        mlflow.end_run()
        
        st.success("✅ Pseudo Labelling hoàn tất!")
        st.markdown(f"[🔗 Truy cập MLflow trên DAGsHub]({st.session_state['mlflow_url']})")


def Semi_supervised():
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
    st.markdown(" ### 🖊️ MNIST NN & Semi-supervised App")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Tổng quan", 
    "Tải dữ liệu",
    "Chia dữ liệu",
    "Huấn luyện", 
    "Thông tin huấn luyện"])

    with tab1: 
        tong_quan()
    with tab2: 
        upload_data()
    with tab3: 
        split_data()
    with tab4: 
        pseudo_labelling()
    with tab5:
        show_mlflow_experiments()
        
def run():
    Semi_supervised() 

if __name__ == "__main__":
    run()
