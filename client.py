import sounddevice as sd
import numpy as np
import socket
import pickle


# デバイス設定 ##############################################################################################################################
#device_list = sd.query_devices() # デバイス一覧
#print(device_list)
sd.default.device = [1, 6] # Input, Outputデバイス指定 [1, 6]


# クライアント操作(WIFI通信) ################################################################################################################
def operation(select):

    HOST_NAME = "192.168.4.1"
    PORT = 5000
    TEXT = "1"

    if(select == 0):
        print("            Process : No Operation")
    elif(select == 1):
        print("            Process : 1")
        HOST_NAME = "192.168.4.1"
        PORT = 5000
        TEXT = "1"
    elif(select == 2):
        print("            Process : 2")
        HOST_NAME = "192.168.4.1"
        PORT = 5000
        TEXT = "2"
    elif(select == 3):
        print("            Process : 3")
        HOST_NAME = "127.0.0.3"
        PORT = 5000
        TEXT = "3"
    elif(select == 4):
        print("            Process : 4")
        HOST_NAME = "127.0.0.4"
        PORT = 5000
        TEXT = "4"

    if(select != 0):
            sk_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   #ソケット作成(TCP)
            #sk_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    #ソケット作成(UDP)
            try:
                sk_client.connect((HOST_NAME, PORT))                    #TCP
                sk_client.send(TEXT.encode("utf-8"))                    #TCP
                #send = sk_client.sendto("TEST".encode("utf-8"), (HOST_NAME, PORT))     #UDP
                sk_client.close()
                #print("Connection ( IP :",HOST_NAME,")")               #UDP
                print("Connection Successful ( IP :",HOST_NAME,")")     #TCP
            except socket.error:
                sk_client.close() 
                print("Connection Faild ( IP :",HOST_NAME,")")          #TCP

    print("")

    return None


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


# タイミングとそのスぺクトルの抽出 #############################################################################################################
def exp_timing(data):
    global TIME, average, maxdata

    count = 0
    thresh_over = 0
    time = 3
    Timing = np.zeros((TIME))

    for i in range(TIME):
        average[i] = np.average(data[i])

        if (average[i] >= 0.000030 and time >= 5) :
            #print(average[i])
            Timing[count] = i
            maxdata[count] = max(data[i])
            count += 1
            time = 0
        time += 1   

    print("              Count :", count)   
    print("               Time :", Timing[0:count])  
    print("                Max :", np.round(maxdata[0:count], 4))  
    print("")

    return count, Timing, maxdata,

# 回数判定関数 ##############################################################################################################################
def count(data):
    global TIME

    C, T, M = exp_timing(data)          #波形の山のタイミングを抽出

    filename = 'model_sample.pickle'    #モデルの読み込み
    clf = pickle.load(open(filename, 'rb')) 

    print("- 4.Prediction --------------------------------")
    true_count = 0
    for i in range(C) :
        dataset = data[int(T[i])].reshape(-1,1050) / M[i] * 0.999999999999
        y_pred = clf.predict(dataset)
        print("                    :", y_pred)
        if (y_pred == 1): true_count += 1
    print("")

    print("- 5.Result ------------------------------------")
    print("         True Count :", true_count) 

    C = 0

    return C #true_count


# 更新関数 ##################################################################################################################################
def update_plot():
    global plotdata, Framesize, Time, wait, detect, record, recording_data, TIME

    data = round(np.abs(np.max(plotdata[42963:44099])),3)


    #時間経過
    Time += 1
    if (Time >= 3 and data < 0.1 and record == False):
        wait = False

    #録音
    if (Time < TIME and record == True):
        spectrum = fourier()                #短時間フーリエ変換
        recorded_spectrum[Time] = abs(spectrum[0][0:int(Framesize / 2)].real)

    if (Time >= TIME and record == True):
        Time = 0
        record = False
        print("- 3.Recorded ----------------------------------")
        select = count(recorded_spectrum)   #回数判定関数
        operation(select)                   #WIFI操作
        print("-----------------------------------------------")
        print("")

    #録音開始合図
    if (data < 0.01 and wait == False and record == False and detect == True):
        Time = 0
        record = True
        detect = False
        print("- 2.Recording ---------------------------------")
        print("")

    #音検出
    if (data > 0.01 and wait == False and record == False):
        print("- 1.Detected ----------------------------------")
        wait = True
        detect = True
        Time = 0
        print("")

    if(record == False):
        None

    return None,

# パラメータ ################################################################################################################################
downsample = 1
Framesize = 2100
fsample = 44100
TIME = 48                                           #録音時間
length = int(1000 * 44100 / (1000 * downsample))

plotdata = np.zeros((length))
recorded_spectrum = np.zeros((TIME, int(Framesize / 2)))
average = np.zeros((TIME))
maxdata = np.zeros((TIME))

window = np.hamming(Framesize)
freq = np.fft.fftfreq(Framesize, d = 1 / length)

Time = 0
detect = False
wait = False
record = False

# 収音クラス ################################################################################################################################
stream = sd.InputStream(channels = 1,     
               dtype = 'float32',
               callback = callback   #callbackにデータを送る
               )

with stream:
    while(True):
        sd.sleep(1)