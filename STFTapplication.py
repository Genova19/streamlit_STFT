import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import streamlit as st
import scipy.signal
from PIL import Image

st.title('STFT Spectrogram APP')

uploaded_file = st.sidebar.file_uploader("CSVファイルのアップロード", type="csv")

# uploadファイルが存在するときだけ、csvファイルの読み込みがされる。
if uploaded_file is not None:
	# コメント行をスキップして読み込んでくれる
    df = pd.read_csv(uploaded_file, header=None,  encoding="ms932", comment="#")
    hist=df.loc[1:]
    hist.columns=df.loc[0]

i=0

st.sidebar.write("""
## データ解析範囲
""")
target = st.sidebar.selectbox('解析物理量の選択', df.loc[0])
fs = st.sidebar.number_input('サンプリング周波数[Hz]', 20,1000000,1000,step=1)

wave=hist[target].astype(float)
tmax=len(wave)/fs
xmin, xmax = st.sidebar.slider('横軸範囲を指定してください。',0.0,tmax,(0.0, tmax))
# tt=hist['Time'].astype(float)
analysis_min=int(xmin*fs)
analysis_max=int(xmax*fs)
wave_analysis=wave[analysis_min:analysis_max-analysis_min+1]
# tt_analysis=tt[analysis_min:analysis_max-analysis_min+1]


st.sidebar.write("""
## STFT Parameter
""")
wfs = ['hann', 'boxcar', 'hamming', 'blackman']
wf= st.sidebar.selectbox('窓関数を選択してください', wfs)
npsg = st.sidebar.slider('STFTのデータ数(perseg)を指定してください。',16,1024,256)
novl = st.sidebar.slider('Overlapを指定してください。',0,256,0)





with st.form(key='user'):

    filename: str =st.text_input('ファイル名を入力（拡張子なし）')
    dxmin, dxmax = st.slider('横軸表示範囲',xmin,xmax,(xmin, xmax))
    dymin, dymax = st.slider('縦軸表示範囲',1,int(fs/2),(10, int(fs/2)), step=1)
    y_axis_log = st.checkbox('縦軸対数', value=True)
    dzmin, dzmax = st.slider('カラーバー範囲 (10^x: 指数を指定)',-6,2,(-5, 1), step=1)
    cmaps =['jet','viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
            'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
            'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','flag', 'prism', 'ocean', 'gist_earth', 'terrain',
            'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
            'turbo', 'nipy_spectral', 'gist_ncar']
    cmap= st.selectbox('カラーマップを選択してください', cmaps)
    but_sub = st.form_submit_button(label='Draw graphs')



# but = st.button('Draw graph')

if but_sub:
    if filename =="":
        st.error('ファイル名を入れてください。')
    else:
        fle_PNG="./" + filename + "_mag.png"
        fle_mCSV="./" + filename + "_mag.csv"
        fle_rCSV="./" + filename + "_raw.csv"

        frq, t, rPxx = scipy.signal.stft(wave_analysis, fs=fs, window=wf, nperseg=npsg, noverlap=novl)
        # 周波数、時間、強さの3つの情報が帰ってくる
        df_rawPxx=pd.DataFrame(rPxx)
        df_rawPxx.columns = t
        df_rawPxx.index = frq
        rcsv=df_rawPxx.to_csv()

        Pxx = (np.abs(rPxx))*2
        df_magPxx=pd.DataFrame(Pxx)
        df_magPxx.columns = t
        df_magPxx.index = frq
        # df_magPxx.to_csv(fle_mCSV)
        mcsv=df_magPxx.to_csv()

        # プロットエリアの用意
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)

        # 生波形
        ax1.plot(np.arange(len(wave_analysis))/fs, wave_analysis)
        ax1.axis('tight')
        ax1.set_ylabel('Raw wave')
        # ax1.set_xlim(0, len(wave_analysis)/fs) # 時間範囲を設定
        ax1.set_xlim(dxmin, dxmax) # 時間範囲を設定
        # ax1.set_ylim(-1, 10)

        # STFT のコンター(contour) 図
        vmin=10**dzmin
        vmax=10**dzmax
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        img2 = ax2.imshow(Pxx, origin='lower', cmap=cmap, extent=(0, np.max(t), 0, fs/2), norm=norm)
        ax2.axis('tight')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time[s]')
        # ax2.set_xlim(10,30) # 時間範囲を設定したいとき
        ax2.set_xlim(dxmin, dxmax) # 時間範囲を設定
        ax2.set_ylim(dymin,dymax) # 周波数範囲を設定したいとき
        if y_axis_log :
            ax2.set_yscale('log')


        # カラーバーを位置調整しつつ追加する。
        # 1.05 で右にはみ出させている。
        axins = inset_axes(ax2,
            width="2%",  # width = 2% of parent_bbox width
            height="100%",  # height : 100%
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax2.transAxes,
            borderpad=0,
            )
        cbar2 = fig.colorbar(img2, cax=axins)

        # 描画マージン（画面の Subplot configuartion tool で調整できるもの）
        # 最終的な描画サイズに合わせて数値調整する必要がある。
        plt.subplots_adjust(left=0.11, bottom=0.095, right=0.87, top=0.98, wspace=0, hspace=0.15)
        #fig.tight_layout()

        # 描画
        plt.show()
        fig.savefig(fle_PNG)
        image = Image.open(fle_PNG)
        st.write('# STFT results')
        st.image(image)

        href2 = f'<a href="data:application/octet-stream;{mcsv}" download="{filename}_mag.csv">Download Link</a>'
        st.markdown(f"STFT result (magnitude) CSVファイルのダウンロード:  {href2}", unsafe_allow_html=True)
        href3 = f'<a href="data:application/octet-stream;{rcsv}" download="{filename}_raw.csv">Download Link</a>'
        st.markdown(f"STFT result (raw) CSVファイルのダウンロード:  {href3}", unsafe_allow_html=True)

        plt.close()
