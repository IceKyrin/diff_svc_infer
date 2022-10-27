import json
import logging

import soundfile

from diffsvc import infer_tool
from diffsvc.infer_tool import Svc
from wav_temp import merge

logging.getLogger('numba').setLevel(logging.WARNING)
record = json.load(open("batch.json", 'r', encoding='utf-8'))

# 工程文件夹名，训练时用的那个
project_name = "yilanqiu"
model_path = f'./checkpoints/{project_name}/model_ckpt_steps_98000.ckpt'

# 新建batch文件夹，批量放置wavs
clean_names = infer_tool.get_end_file("./batch", "wav")
clean_names = [i.split("/")[-1][:-4] for i in clean_names]
trans = [-6]  # 音高调整，支持正负（半音），填一个就行
# 加速倍数
accelerate = 50

# 下面不动
infer_tool.mkdir(["./batch", "./results"])

input_wav_path = "./wav_temp/input"
out_wav_path = "./wav_temp/output"
cut_time = 60

svc_model = Svc(project_name, model_path)
infer_tool.fill_a_to_b(trans, clean_names)
infer_tool.mkdir([input_wav_path, out_wav_path])

# 清除缓存文件
infer_tool.del_temp_wav(input_wav_path)
rec = 0
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"./batch/{clean_name}.wav"
    infer_tool.del_temp_wav("./wav_temp")
    out_audio_name = clean_name
    infer_tool.cut_wav(raw_audio_path, out_audio_name, input_wav_path, cut_time)

    count = 0
    file_list = infer_tool.get_end_file(input_wav_path, "wav")
    for file_name in file_list:
        file_name = file_name.split("/")[-1]
        raw_path = f"{input_wav_path}/{file_name}"
        out_path = f"{out_wav_path}/{file_name}"

        audio = svc_model.infer(raw_path, key=tran, acc=accelerate, use_pe=True, use_gt_mel=False, add_noise_step=500)
        soundfile.write(out_path, audio, 24000, 'PCM_16')

        count += 1
    merge.run(out_audio_name, f"")
    record[clean_name] = tran
    rec += 1
    # 清除缓存文件
    infer_tool.del_temp_wav("./wav_temp/output")
    if rec % 10 == 0:
        with open('batch.json', 'w', encoding='utf8') as f2:
            json.dump(record, f2, ensure_ascii=False, indent=2)

with open('batch.json', 'w', encoding='utf8') as f2:
    json.dump(record, f2, ensure_ascii=False, indent=2)
