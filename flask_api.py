import io
import logging

import librosa
import soundfile
from flask import Flask, request, send_file
from flask_cors import CORS

from diffsvc.infer_tool import Svc

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # http获得wav文件并转换
    input_wav_path = "./wav_temp/http_in.wav"
    with open(input_wav_path, "wb") as f:
        f.write(wave_file.read())

    # 模型推理
    out_audio = svc_model.infer(input_wav_path, key=f_pitch_change, acc=accelerate, use_pe=True, use_gt_mel=False,
                                add_noise_step=500)
    tar_audio = librosa.resample(out_audio, 24000, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 每个模型和config是唯一对应的
    # 工程文件夹名，训练时用的那个
    project_name = "yilanqiu"
    model_path = f'./checkpoints/{project_name}/model_ckpt_steps_66000.ckpt'
    # 加速倍数
    accelerate = 50
    svc_model = Svc(project_name, model_path)
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
