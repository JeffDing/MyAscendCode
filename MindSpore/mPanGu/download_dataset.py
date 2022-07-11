import os
import time
import logging
import moxing as mox
import sys
##
##

# BUCKET_DIR0 = "obs://datasets/V1-sample300-bpe-1024/"
# BUCKET_DIR1 = "obs://datasets/V1-sample300-baike-bpe-1024/"
# BUCKET_DIR = 'obs://datasets/V1-finetune-bpe-1024/'
# LOCAL_PATH0 = "/cache/V1-sample300-bpe-1024"
# LOCAL_PATH1 = "/cache/V1-sample300-baike-bpe-1024"

# BUCKET_DIR = 'obs://datasets/V1-finetune-2_7g-bpe-1024/'
# LOCAL_PATH = "/cache/V1-finetune-2_7g-bpe-1024"

# BUCKET_DIR = 'obs://datasets/V1-language-50_2g-bbpe-1024/'
# LOCAL_PATH = "/cache/V1-language-50_2g-bbpe-1024"
# BUCKET_DIR = 'obs://datasets/v1-5multilingual-zh-arabic-T2-bbpe-1024/'
# LOCAL_PATH = "/cache/v1-5multilingual-zh-arabic-T2-bbpe-1024"
# BUCKET_DIR = 'obs://datasets/v1-ChineseLyricsGen0729-150G-bbpe-1024/'
# LOCAL_PATH = "/cache/v1-ChineseLyricsGen0729-150G-bbpe-1024"
# BUCKET_DIR = 'obs://datasets/v1-ChineseLyricsGen0729-90M-bpe-1024/'
# LOCAL_PATH = "/cache/v1-ChineseLyricsGen0729-90M-bpe-1024"
# BUCKET_DIR = 'obs://datasets/v1-ChineseEnglish-100g-bpe-1024/'
# LOCAL_PATH = "/cache/v1-ChineseEnglish-100g-bpe-1024"
BUCKET_DIR = ''
LOCAL_PATH = "/cache/v1-ChineseEnglish-130g-bpe-1024"
LEN_1024 = 120
LEN_2048 = 3600

class DatasetDownloader:
    def __init__(self, bucket_dir, retry=3, retry_time=30, interval_time=30):
        self.bucket_dir = bucket_dir
        self.retry = retry
        self.retry_time = retry_time
        self.interval_time = interval_time

    def down_dataset(self, local_path):
        rank_id = int(os.getenv('RANK_ID'))

        if rank_id % 8 == 0:
            device_id = rank_id // 8
            files = os.listdir(local_path)
            success = False
            # FIx, log path 
            log_dir = os.path.join("", "download_100g_bbpe_log")
            if not mox.file.exists(log_dir):
                mox.file.mk_dir(log_dir)

            except_file_path = os.path.join(log_dir, f"except_download_device_{device_id}.log")
            success_file_path = os.path.join(log_dir, f"success_download_device_{device_id}.log")
            except_info = ""
            if files:
                # change number to the target
                if len(files) == LEN_1024:
                    mox.file.append(success_file_path, f"device_{device_id}: already downloading {self.bucket_dir} to "
                                    f"{local_path} succeed.\n")
                    logging.info(f"device_{device_id} dataset already  exists.")
                    return
                else:
                    logging.info(f"device_{device_id} remove files")
                    for data_file in files:
                        os.remove(os.path.join(local_path, data_file))

            # sleep due to restriction of obs
            if device_id > 5:
                while (not mox.file.exists(os.path.join(log_dir, f"success_download_device_{device_id-6}.log"))
                       and not mox.file.exists(os.path.join(log_dir, f"except_download_device_{device_id-6}.log"))):
                    time.sleep((device_id % 8 + 1) * 4)

            logging.info(f"device_{device_id}: start downloading {self.bucket_dir} to {local_path}.")
            for i in range(self.retry + 1):
                try:
                    start = time.time()
                    mox.file.copy_parallel(self.bucket_dir, local_path)
                    end = time.time()
                    success = True
                    logging.info(
                        f"device_{device_id}: downloading {self.bucket_dir} to {local_path} cost {end - start}s.")
                    break
                except Exception as e:
                    if i < self.retry:
                        logging.info(e.__str__() + f" device_{device_id}: downloading {self.bucket_dir} to {local_path}"
                                                   f" failed: retry {i + 1}/{self.retry}.")
                        time.sleep(self.retry_time)
                    else:
                        except_info = e.__str__()

            if not success:
                mox.file.append(except_file_path, f"{except_info}. device_{device_id}: downloading {self.bucket_dir} to "
                                                  f"{local_path} failed.\n")
            else:
                mox.file.append(success_file_path, f"device_{device_id}: downloading {self.bucket_dir} to "
                                                   f"{local_path} succeed.\n")


if __name__ == "__main__":
    downloader = DatasetDownloader(BUCKET_DIR)
    downloader.down_dataset(LOCAL_PATH)
    print("DONE")