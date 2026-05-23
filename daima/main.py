import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
import pandas as pd
import time  # 用于防重复打卡

# 全局初始化
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = None


# 数据库操作
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students 
                 (student_id TEXT PRIMARY KEY, name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (id INTEGER PRIMARY KEY, student_id TEXT, date TEXT, time TEXT, status TEXT)''')
    conn.commit()
    conn.close()


def insert_student(student_id, name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO students VALUES (?, ?)", (student_id, name))
    conn.commit()
    conn.close()


def get_name_by_id(student_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "未知"


def log_attendance(student_id):
    now = datetime.now()
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO attendance VALUES (NULL, ?, ?, ?, '出勤')",
              (student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    conn.commit()
    conn.close()


# 训练模型
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        filename = os.path.split(image_path)[-1]
        student_id = filename.split(".")[0]
        faces = face_cascade.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(student_id)
    return face_samples, ids


def train_model():
    global recognizer
    if not os.path.exists('dataset') or not os.listdir('dataset'):
        messagebox.showerror("错误", "请先完成人脸注册！")
        return
    faces, ids = get_images_and_labels('dataset')
    if len(faces) == 0:
        messagebox.showerror("错误", "未检测到有效人脸数据！")
        return
    unique_ids = list(set(ids))
    id_to_label = {uid: idx for idx, uid in enumerate(unique_ids)}
    label_ids = [id_to_label[i] for i in ids]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(label_ids))
    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')

    with open('trainer/id_map.txt', 'w', encoding='utf-8') as f:
        for uid, label in id_to_label.items():
            f.write(f"{label}:{uid}\n")

    messagebox.showinfo("成功", f"模型训练完成！共训练 {len(unique_ids)} 名学生，{len(faces)} 张人脸。")


# 注册窗口
class RegisterWindow:
    def __init__(self, master):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title("人脸注册")
        self.top.geometry("900x700")
        tk.Label(self.top, text="学号：", font=("微软雅黑", 12)).pack(pady=5)
        self.id_entry = tk.Entry(self.top, font=("微软雅黑", 12), width=30)
        self.id_entry.pack()
        tk.Label(self.top, text="姓名：", font=("微软雅黑", 12)).pack(pady=5)
        self.name_entry = tk.Entry(self.top, font=("微软雅黑", 12), width=30)
        self.name_entry.pack()
        tk.Button(self.top, text="开始采集（请正对摄像头）", font=("微软雅黑", 12), bg="#4CAF50", fg="white",
                  command=self.start_capture).pack(pady=10)
        self.cam_label = tk.Label(self.top)
        self.cam_label.pack()
        self.cap = None
        self.count = 0
        self.student_id = None
        self.name = None

    def start_capture(self):
        self.student_id = self.id_entry.get().strip()
        self.name = self.name_entry.get().strip()
        if not self.student_id or not self.name:
            messagebox.showerror("错误", "学号和姓名不能为空！")
            return
        insert_student(self.student_id, self.name)
        os.makedirs('dataset', exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        self.count = 0
        self.capture_loop()

    def capture_loop(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if self.count < 30:
                    face_img = frame[y:y + h, x:x + w]
                    cv2.imwrite(f"dataset/{self.student_id}.{self.count + 1}.jpg", face_img)
                    self.count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        if self.count < 30:
            self.top.after(10, self.capture_loop)
        else:
            self.cap.release()
            messagebox.showinfo("完成", f"学生 {self.name} 人脸采集完成（30张）！")
            self.top.destroy()


# 考勤识别窗口
class AttendanceWindow:
    def __init__(self, master):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title("实时考勤")
        self.top.geometry("1000x700")
        self.cam_label = tk.Label(self.top)
        self.cam_label.pack()
        self.status_label = tk.Label(self.top, text="等待识别...", font=("微软雅黑", 14), fg="blue")
        self.status_label.pack(pady=10)
        tk.Button(self.top, text="开始考勤", font=("微软雅黑", 12), command=self.start_attendance).pack(pady=5)
        tk.Button(self.top, text="停止", font=("微软雅黑", 12), command=self.stop).pack(pady=5)
        self.cap = None
        self.running = False
        self.last_log = {}  # 防重复打卡字典
        global recognizer
        if os.path.exists('trainer/trainer.yml'):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer/trainer.yml')
            self.id_map = {}
            if os.path.exists('trainer/id_map.txt'):
                with open('trainer/id_map.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        label, uid = line.strip().split(':', 1)
                        self.id_map[int(label)] = uid
        else:
            messagebox.showwarning("警告", "请先训练模型！")

    def start_attendance(self):
        if not os.path.exists('trainer/trainer.yml'):
            messagebox.showerror("错误", "请先训练模型！")
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.recognize_loop()

    def recognize_loop(self):
        if not self.running or self.cap is None: return
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                label, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 80:
                    student_id = self.id_map.get(label, "未知")
                    name = get_name_by_id(student_id)

                    # 同一人60秒内只记录一次
                    current_time = time.time()
                    if student_id not in self.last_log or current_time - self.last_log[student_id] > 60:
                        log_attendance(student_id)
                        self.last_log[student_id] = current_time
                        self.status_label.config(text=f"识别成功：{name}（已记录）", fg="green")

                    # 显示文字
                    cv2.putText(frame, f"{name} 已到", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "未知", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        self.top.after(10, self.recognize_loop)

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()
        self.top.destroy()


# 查询窗口
class QueryWindow:
    def __init__(self, master):
        self.master = master
        self.top = tk.Toplevel(master)
        self.top.title("考勤查询")
        self.top.geometry("1100x700")
        columns = ("学号", "姓名", "日期", "时间", "状态")
        self.tree = ttk.Treeview(self.top, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        tk.Button(self.top, text="刷新数据", command=self.load_data).pack(pady=5)
        tk.Button(self.top, text="导出为Excel", command=self.export_excel).pack(pady=5)
        self.load_data()

    def load_data(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT s.student_id, s.name, a.date, a.time, a.status "
                               "FROM attendance a JOIN students s ON a.student_id = s.student_id", conn)
        conn.close()
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=tuple(row))

    def export_excel(self):
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT s.student_id, s.name, a.date, a.time, a.status "
                               "FROM attendance a JOIN students s ON a.student_id = s.student_id", conn)
        conn.close()
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("成功", f"已导出到：{file_path}")


# 主界面
class MainApp:
    def __init__(self):
        init_db()
        self.root = tk.Tk()
        self.root.title("基于OpenCV的人脸识别考勤系统")
        self.root.geometry("800x600")
        tk.Label(self.root, text="基于OpenCV的人脸识别考勤系统", font=("微软雅黑", 20, "bold")).pack(pady=20)

        btn_style = {"font": ("微软雅黑", 14), "width": 25, "height": 2}
        tk.Button(self.root, text="1. 人脸注册", bg="#2196F3", fg="white", **btn_style,
                  command=lambda: RegisterWindow(self.root)).pack(pady=8)
        tk.Button(self.root, text="2. 训练模型", bg="#FF9800", fg="white", **btn_style,
                  command=train_model).pack(pady=8)
        tk.Button(self.root, text="3. 实时考勤", bg="#4CAF50", fg="white", **btn_style,
                  command=lambda: AttendanceWindow(self.root)).pack(pady=8)
        tk.Button(self.root, text="4. 考勤查询与导出", bg="#9C27B0", fg="white", **btn_style,
                  command=lambda: QueryWindow(self.root)).pack(pady=8)
        tk.Button(self.root, text="退出系统", bg="#f44336", fg="white", **btn_style,
                  command=self.root.quit).pack(pady=20)

        tk.Label(self.root, text="提示：先注册 → 训练 → 考勤 → 查询", fg="gray").pack(side="bottom", pady=20)
        self.root.mainloop()


if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('trainer', exist_ok=True)
    MainApp()