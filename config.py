# my_project/my_project/config.py

import warnings
import torch

class Config:
    USE_GPU = True
    USE_RANDOM_SPLIT = False
    USE_DATAPARALLEL = True
    SEED = 42

    # ====== 資料集相關路徑與檔案名稱 ======
    IMAGE_DIR = '5_5_cactus_48x29_v1/Image'
    LABEL_FILE = '5_5_cactus_48x29_v1/label.txt'
    MODEL_PATH = '5_5_cactus_48x29_vit_b16.pth'

    # ====== 圖片大小 ======
    # 若使用自訂 CNN (灰階): 預設 48x29
    # 也可以在 transforms.py 動態決定
    IMG_HEIGHT = {5: 48, 20: 64, 60: 96}
    IMG_WIDTH = {5: 29, 20: 60, 60: 180}

    # ====== 訓練參數 ======
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 128

    # ====== 模型種類 ======
    MODEL_TYPE = 'custom'  # 'custom', 'resnet18', 'vit_b16', ...

    EARLY_STOPPING_PATIENCE = 2
    MIN_DELTA = 1e-5

def setup_gpu():
    """
    選擇可用的 GPU (若有)，否則使用 CPU。
    """
    warnings.filterwarnings('ignore')
    torch.manual_seed(Config.SEED)

    import os, re
    if Config.USE_GPU:
        def query_gpu(qargs=[]):
            qargs = ['index', 'gpu_name', 'memory.free'] + qargs
            cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
            results = os.popen(cmd).readlines()
            return results

        def select_gpu(results, thres=4096):
            available = []
            try:
                for i, line in enumerate(results):
                    mem_free = int(re.findall(r'(.*), (.*?) MiB', line)[0][-1])
                    if mem_free > thres:
                        available.append(i)
                return available
            except:
                return ''
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(gpu) for gpu in select_gpu(query_gpu())]
        )

    if not torch.cuda.is_available() and Config.USE_GPU:
        raise RuntimeError("此程式需要 GPU 環境才能運行，請確認 CUDA 是否安裝正確。")

    return torch.device('cuda:0' if torch.cuda.is_available() and Config.USE_GPU else 'cpu')
