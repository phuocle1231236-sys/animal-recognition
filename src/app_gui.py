import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from model_pipeline import AnimalPredictor

class App:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("HỆ THỐNG NHẬN DẠNG ĐỘNG VẬT V3")
        self.root.geometry("500x600")

        self.label_title = tk.Label(root, text="Phân loại 10 loài động vật", font=("Arial", 16, "bold"))
        self.label_title.pack(pady=10)

        self.btn_select = tk.Button(root, text="Chọn ảnh dự đoán", command=self.select_image, 
                                   font=("Arial", 12), bg="#4CAF50", fg="white", padx=20)
        self.btn_select.pack(pady=10)

        self.canvas = tk.Label(root) 
        self.canvas.pack(pady=10)

        self.result_text = tk.Label(root, text="Kết quả: Đang chờ...", font=("Arial", 14), fg="blue")
        self.result_text.pack(pady=20)

    def select_image(self):
        path = filedialog.askopenfilename()
        if not path: return

        # 1. Hiển thị ảnh lên giao diện
        img_display = Image.open(path).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_display)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

        # 2. Dự đoán bằng Pipeline AI đã đóng gói
        animal_vn, conf = self.predictor.predict(path)

        # 3. Hiển thị kết quả
        self.result_text.config(text=f"KẾT QUẢ: {animal_vn}\nĐộ tin cậy: {conf:.2%}")

if __name__ == "__main__":
    # Khởi tạo mô hình trước khi mở giao diện để tránh bị crash giữa chừng
    predictor = AnimalPredictor()
    
    root = tk.Tk()
    app = App(root, predictor)
    root.mainloop()