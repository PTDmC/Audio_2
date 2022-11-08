import numpy as np
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import csv


# デバイス設定 ##############################################################################################################################
device_list = sd.query_devices() # デバイス一覧
#print(device_list)
sd.default.device = [1, 4] # Input, Outputデバイス指定

# リアルタイム収音 ##########################################################################################################################
def callback(indata, frames, time, status):         #indataには0.026秒分のデータが格納される
    global plotdata
    data = indata[::downsample, 0]                  #[::N] indataの要素をN(downsample)個ずつ飛ばして取り出す。　0はチャンネル。
    shift = len(data)                               #dataの要素数を取得
    plotdata = np.roll(plotdata, -shift, axis=0)    #plotdataの要素をdata(shift)の要素数分マイナスシフトする。axis=0は行方向のシフト。
    plotdata[-shift:] = data                        #後ろからshift番目までにdataを代入(plotを右から流れるようにするため)
    update_plot()


# フーリエ変換 ##############################################################################################################################
def fourier():
    global plotdata, Framesize, window
    Frame = plotdata[-Framesize:] * window
    F = np.fft.fft(Frame)
    F = F / (Framesize / 2)                             # フーリエ変換の結果を正規化
    F = F * (Framesize / sum(window))                   # 窓関数による振幅補正
    FFT_result = 20 * np.log10(np.abs(F) + 1e-18)       # 振幅スペクトル

    return F, #FFT_result

# 学習用スペクトルの抽出 ####################################################################################################################
def explicit(data):
    global TIME, average, maxdata

    for i in range(TIME):
        average[i] = np.average(data[i])
        maxdata[i] = max(data[i])
    maxtime = np.argmax(average)
    mdata = maxdata[maxtime]
    print(maxtime, mdata)

    return maxtime, mdata

# 更新関数 ##################################################################################################################################
def update_plot():
    global plotdata, Framesize, Time, wait, detect, record, recorded_spectrum, TIME

    data = round(np.abs(np.max(plotdata[42963:44099])),3)

    #時間経過
    Time += 1
    if (Time >= 3 and data < 0.1 and record == False):
        wait = False

    #録音
    if (Time < TIME and record == True):
        spectrum = fourier()
        recorded_spectrum[Time] = abs(spectrum[0][0:int(Framesize / 2)].real)

    if (Time >= TIME and record == True):
        Time = 0
        record = False
        T, M = explicit(recorded_spectrum)

        writedata = recorded_spectrum[T] / M * 0.999999999999
        judge = input("This Sample is ClapSound?(1 or 0) :")
        writedata = np.insert(writedata, 0, int(judge))

        with open('C:/Users/siura/source/repos/Project_2022_2/Project_2022_2/learning_data/data.csv', 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow(writedata)
        print("-recorded----------------------------------")
        print("Wait 2[sec]")
        sd.sleep(2 * 1000)
        print("Stand-by")

    #録音開始合図
    if (data < 0.1 and wait == False and record == False and detect == True):
        Time = 0
        record = True
        detect = False
        print("-recording---------------------------------")

    #音検出
    if (data > 0.2 and wait == False and record == False):
        print("-detected----------------------------------")
        wait = True
        detect = True
        Time = 0

    if(record == False):
        None

    return None


# パラメータ ################################################################################################################################
downsample = 1
Framesize = 2100
fsample = 44100
TIME = 72                                           #録音時間
length = int(1000 * 44100 / (1000 * downsample))

plotdata = np.zeros((length))
recorded_spectrum = np.zeros((TIME, int(Framesize / 2)))
average = np.zeros((TIME))
maxdata = np.zeros((TIME))

window = np.hamming(Framesize)
freq = np.fft.fftfreq(Framesize, d = 1 / length)

Time = 0
detect = False
wait = True
record = False


# 収音クラス ################################################################################################################################
stream = sd.InputStream(channels = 1,     
               dtype = 'float32',
               callback = callback   #callbackにデータを送る
               )

with stream:
    while(True):
        sd.sleep(1)
