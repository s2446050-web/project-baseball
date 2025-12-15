import streamlit as st
import cv2
import numpy as np
import tempfile
import av
import math
import random
import os
from collections import deque, defaultdict
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ==========================================
# 1. 解析用ロジック
# ==========================================
class CentroidTracker:
    def __init__(self, maxDisappeared=5, maxDistance=100):
        self.nextObjectID = 0
        self.objects = dict(); self.disappeared = dict()
        self.maxDisappeared = maxDisappeared; self.maxDistance = maxDistance
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid; self.disappeared[self.nextObjectID] = 0; self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]; del self.disappeared[objectID]
    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects
        if len(self.objects) == 0:
            for c in inputCentroids: self.register(c)
            return self.objects
        objectIDs = list(self.objects.keys())
        objectCentroids = [self.objects[oid] for oid in objectIDs]
        D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype="float")
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(inputCentroids): D[i, j] = math.hypot(oc[0]-ic[0], oc[1]-ic[1])
        rows, cols = D.shape
        assignedRows = set(); assignedCols = set(); all_pairs = []
        for i in range(rows):
            for j in range(cols): all_pairs.append((D[i, j], i, j))
        all_pairs.sort(key=lambda x: x[0])
        pairs = []
        for dist, i, j in all_pairs:
            if i in assignedRows or j in assignedCols or dist > self.maxDistance: continue
            assignedRows.add(i); assignedCols.add(j); pairs.append((i, j))
        matchedRows = set([p[0] for p in pairs]); matchedCols = set([p[1] for p in pairs])
        for (row, col) in pairs:
            objectID = objectIDs[row]; self.objects[objectID] = inputCentroids[col]; self.disappeared[objectID] = 0
        for i in range(rows):
            if i not in matchedRows:
                objectID = objectIDs[i]; self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
        for j in range(cols):
            if j not in matchedCols: self.register(inputCentroids[j])
        return self.objects

def random_color(seed):
    random.seed(seed)
    return (int(random.random()*200)+30, int(random.random()*200)+30, int(random.random()*200)+30)

class BallTracker:
    def __init__(self):
        # MOG2設定
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.tracker = CentroidTracker(maxDisappeared=5, maxDistance=100)
        self.trails = defaultdict(lambda: deque(maxlen=64))
        self.colors = dict()

    def process_frame(self, frame):
        try:
            # 画像処理
            fgmask = self.fgbg.apply(frame)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, k, iterations=1)
            fgmask = cv2.GaussianBlur(fgmask, (3,3), 0)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for c in cnts:
                if cv2.contourArea(c) < 50: continue # 小さいボールも検知
                (x,y), radius = cv2.minEnclosingCircle(c)
                if radius < 2: continue
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                detections.append((cx, cy))

            objects = self.tracker.update(detections)
            
            active_ids = []
            for oid, centroid in objects.items():
                active_ids.append(oid)
                if oid not in self.colors: self.colors[oid] = random_color(oid+10)
                self.trails[oid].appendleft(centroid)
            
            for oid in list(self.trails.keys()):
                if oid not in active_ids: self.trails[oid].appendleft(None)

            vis = frame.copy()
            
            # 軌道の描画
            drawn = False
            for oid, trail in self.trails.items():
                col = self.colors.get(oid, (255,255,255))
                prev = None
                for i, p in enumerate(trail):
                    if p is None: prev = None; continue
                    pt = (int(p[0]), int(p[1]))
                    if prev is not None:
                        thickness = int(np.sqrt(64 / float(i+1)) * 2)
                        cv2.line(vis, prev, pt, col, thickness)
                        drawn = True
                    prev = pt
                if len(trail) > 0 and trail[0] is not None:
                    cv2.circle(vis, (int(trail[0][0]), int(trail[0][1])), 5, col, -1)
            
            # ★デバッグ表示: 解析が動いているか画面左上に表示
            if not drawn:
                cv2.putText(vis, "Scanning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "TRACKING!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return vis

        except Exception as e:
            # ★エラーが起きたら、隠さずに画面に赤文字で書き込む
            err_msg = f"Error: {str(e)}"
            cv2.putText(frame, err_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Processing Error: {e}") # ログにも残す
            return frame

# ==========================================
# 2. WebRTC設定
# ==========================================
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

class VideoProcessor:
    def __init__(self):
        self.tracker = BallTracker()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result_img = self.tracker.process_frame(img)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    st.set_page_config(page_title="Baseball Tracker Pro", layout="wide")
    st.title("⚾ Baseball Trajectory Cloud Pro")
    
    mode = st.sidebar.radio("モード選択", ("Webカメラ (リアルタイム)", "動画ファイルアップロード"))

    if mode == "Webカメラ (リアルタイム)":
        st.info("カメラを起動し、映像の左上に文字（Scanning... または Error）が出るか確認してください。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Camera 1")
            webrtc_streamer(
                key="cam1",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                rtc_configuration=RTC_CONFIGURATION,
            )

    elif mode == "動画ファイルアップロード":
        st.markdown("### 📂 動画ファイル解析")
        file1 = st.file_uploader("動画1", type=["mp4", "mov"])
        file2 = st.file_uploader("動画2", type=["mp4", "mov"])
        
        if st.button("解析スタート") and (file1 or file2):
            tracker1 = BallTracker()
            tracker2 = BallTracker()
            
            # 一時ファイル
            tpath1 = tpath2 = None
            if file1:
                tfile1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile1.write(file1.read()); tpath1 = tfile1.name
            if file2:
                tfile2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile2.write(file2.read()); tpath2 = tfile2.name
            
            cap1 = cv2.VideoCapture(tpath1) if tpath1 else None
            cap2 = cv2.VideoCapture(tpath2) if tpath2 else None

            col1, col2 = st.columns(2)
            ph1 = col1.empty(); ph2 = col2.empty()
            
            # 保存設定 (修正: avc1 -> mp4v に戻す)
            output_path = "cloud_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            target_w, target_h = 640, 480
            final_w = target_w * 2 if (cap1 and cap2) else target_w
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (final_w, target_h))

            st.text("解析中...")

            while True:
                ret1, frame1 = cap1.read() if cap1 and cap1.isOpened() else (False, None)
                ret2, frame2 = cap2.read() if cap2 and cap2.isOpened() else (False, None)
                if not ret1 and not ret2: break
                
                # リサイズ
                if ret1: frame1 = cv2.resize(frame1, (target_w, target_h))
                if ret2: frame2 = cv2.resize(frame2, (target_w, target_h))
                
                # 解析
                out1 = tracker1.process_frame(frame1) if ret1 else None
                out2 = tracker2.process_frame(frame2) if ret2 else None

                # 表示更新
                if out1 is not None: ph1.image(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
                if out2 is not None: ph2.image(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))

                # 保存
                final_frame = None
                if out1 is not None and out2 is not None: final_frame = np.hstack((out1, out2))
                elif out1 is not None: final_frame = out1
                elif out2 is not None: final_frame = out2
                
                if writer is not None and final_frame is not None:
                    writer.write(final_frame)

            if cap1: cap1.release()
            if cap2: cap2.release()
            writer.release()
            
            # ファイル生成チェック
            if os.path.exists(output_path):
                st.success("解析完了")
                with open(output_path, "rb") as f:
                    st.download_button("📥 動画をダウンロード", f, "result.mp4", "video/mp4")
            else:
                st.error("動画の保存に失敗しました。")

if __name__ == "__main__":
    main()
