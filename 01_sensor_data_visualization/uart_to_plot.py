# -*- coding: utf-8 -*-
"""
@Objective: To collect and analyze data in real-time style
Created on Fri Dec  6 16:19:38 2019

@author: frank
@company: Biologue Co. Ltd.
"""

import sys
import os
import csv
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.ttk import Combobox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import numpy as np
import multiprocessing
import threading
import datetime
import time
from enum import Enum
from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import threading
import playsound
import dlib
import cv2
import bitalino
from bitalino import BITalino
# from scipy.signal import butter, lfilter
# import librosa

ENABLE_ECG = True               # True: Enable ECG Device, False: Disable ECG Device
# True: Enable button used to change state, False: Disable button used to change state
ENABLE_STATE_CHANGE = True
# True: Enable video recording, False: Disable video recording
ENABLE_VIDEO_RECORDING = True
ECG_CONN_TIMEOUT = 10
ACC_MAX = 65536
HEADER_TRAILER = [0X55, 0XBC, 0X66]
ONE_UNIT_LEN = 15

# Log filename setting
# True: Vertical for one axis, False: Not vertical for all axes
CORRECT_ACC_LOCATION = True

# Serial port setting
PARITY = serial.PARITY_NONE
STOPBITS = serial.STOPBITS_ONE
BYTESIZE = serial.EIGHTBITS

# Unit: Hz
DEFAULT_SAMPLING_RATE = 64
DEFAULT_PLOT_SAMPLING_RATE = 10
BAUD_RATE = [110, 300, 600, 1200, 2400, 4800, 9600,
             14400, 19200, 38400, 56000, 57600, 115200]
ACC_LABEL_Y_AXIS = ['Acc_x', 'Acc_y', 'Acc_z']
WEIGHT_RANGE = [i for i in range(35, 131)]
HEIGHT_RANGE = [i for i in range(150, 201)]
TIME_DURATION = [i for i in range(50, 1001)]

# GUI Setting
SETTING_ROW_START = 2
ECG_ROW_START = 10
RUN_ROW_START = 15
STATUS_ROW_START = 20

# For Video Recording
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48
FALG_READCAMERA = True
FLAG_SHOW = True


class Version(Enum):
    DUMMY = 0                               # For not developer use
    PRO = 1                                 # For Developer use


class AppFigurePlot(Frame):
    global plot_log_proc

    def __init__(self, master=None):
        global ecg_flag_notification
        global bcg_flag_notification
        Frame.__init__(self, master)
        self.grid()
        self.com_port = []
        self.location = ''
        self.downsample = 1
        self.output_filename = ''
        self.__get_available_com_port()
        self.fs = 1
        self.is_plot = True
        self.fft_interval = 5                               # Unit: Seconds
        self.start_timestamp = 0
        self.start_log = False
        if ENABLE_ECG == True:
            self.ecg_mac_addr_found = False
            self.ecg_conn_enable = False
            self.n_samples = 200
            self.ecg_mac_addr = ''
            self.__find_ECG_mac_addr()

        if ENABLE_STATE_CHANGE == True:
            ecg_flag_notification = multiprocessing.Value('i', False)
            bcg_flag_notification = multiprocessing.Value('i', False)
            self.flag_clicked = False
            self.flag_delay_sec = 0
        self.initAppWidgets()
        self.__set_window_location()
        self.master.protocol("WM_DELETE_WINDOW", self.__quit)
        if CORRECT_ACC_LOCATION == True:
            self.acc_location = 1
        else:
            self.acc_location = 0

    def initAppWidgets(self):
        self.icon_photo = PhotoImage(file='logo.png')
        self.logo_gif = PhotoImage(file='logo.gif')
        self.logo_gif = self.logo_gif.subsample(2)

        self.master.title('Show and Save pressure by BioLogue')
        self.master.resizable(0, 0)
        self.master.iconphoto(False, self.icon_photo)

        self.LogoLabel = Label(self, image=self.logo_gif,
                               width=100, height=100)
        self.LogoLabel.grid(row=0, column=0, rowspan=2, columnspan=3)

        fontStyle = tkFont.Font(family='Lucida Grande', size=25)
        self.BpmLabel = Label(self, text='心跳', font=fontStyle)
        self.BpmLabel.grid(row=0, column=3)
        self.RpmLabel = Label(self, text='呼吸', font=fontStyle)
        self.RpmLabel.grid(row=0, column=4)
        self.SignalQualityLabel = Label(self, text='狀態', font=fontStyle)
        self.SignalQualityLabel.grid(row=0, column=5)

        fontStyle = tkFont.Font(family='Lucida Grande', size=40)
        self.BpmValue = IntVar()            # For BPM
        self.BpmValueLabel = Label(
            self, textvariable=self.BpmValue, font=fontStyle)
        self.BpmValueLabel.grid(row=1, column=3)
        self.RpmValue = IntVar()
        self.RpmValueLabel = Label(
            self, textvariable=self.RpmValue, font=fontStyle)
        self.RpmValueLabel.grid(row=1, column=4)
        self.SignalQualityValue = IntVar()
        self.SignalQualityValueLabel = Label(
            self, textvariable=self.SignalQualityValue, font=fontStyle)
        self.SignalQualityValueLabel.grid(row=1, column=5)

        if VERSION == Version.PRO:
            self.COMLabel = Label(self, text='COM Port')
            self.COMLabel.grid(row=SETTING_ROW_START + 1, column=0, sticky=W)
            self.COMCombobox = Combobox(
                self, postcommand=self.__update_com_port)
            self.COMCombobox['values'] = self.com_port
            self.COMCombobox.grid(
                row=SETTING_ROW_START + 1, column=1, sticky=W)
            if len(self.com_port) >= 1:
                self.COMCombobox.current(0)
            self.COMCombobox1 = Combobox(
                self, postcommand=self.__update_com_port)
            self.COMCombobox1['values'] = self.com_port
            self.COMCombobox1.grid(
                row=SETTING_ROW_START + 1, column=2, sticky=W)

            self.BaudRateLabel = Label(self, text='Baud Rate')
            self.BaudRateLabel.grid(
                row=SETTING_ROW_START + 1, column=3, sticky=W)
            self.BaudRateCombobox = Combobox(self)
            self.BaudRateCombobox['values'] = BAUD_RATE
            self.BaudRateCombobox.grid(
                row=SETTING_ROW_START + 1, column=4, sticky=W)
            self.BaudRateCombobox.current(12)

            self.SamplingRateLabel = Label(self, text='Sampling_Rate_Hz')
            self.SamplingRateLabel.grid(
                row=SETTING_ROW_START + 2, column=0, sticky=W)
            self.SamplingRateEntry = Entry(self, width=20)
            self.SamplingRateEntry.insert(INSERT, '64')
            self.SamplingRateEntry.grid(
                row=SETTING_ROW_START + 2, column=1, sticky=W)

            self.DownsampleLabel = Label(self, text='Downsample')
            self.DownsampleLabel.grid(
                row=SETTING_ROW_START + 2, column=2, sticky=W)
            self.DownsampleCombobox = Combobox(self)
            self.DownsampleCombobox['values'] = [(i + 1) for i in range(10)]
            self.DownsampleCombobox.grid(
                row=SETTING_ROW_START + 2, column=3, sticky=W)
            self.DownsampleCombobox.current(0)

            self.DurationLabel = Label(self, text='Duration_Seconds')
            self.DurationLabel.grid(
                row=SETTING_ROW_START + 2, column=4, sticky=W)
            self.DurationEntry = Entry(self, width=20)
            self.DurationEntry.insert(INSERT, '120')
            self.DurationEntry.grid(
                row=SETTING_ROW_START + 2, column=5, sticky=W)

            self.PlotIntervalLabel = Label(self, text='Plot_Interval_Second')
            self.PlotIntervalLabel.grid(
                row=SETTING_ROW_START + 3, column=0, sticky=W)
            self.PlotIntervalEntry = Entry(self, width=20)
            self.PlotIntervalEntry.insert(INSERT, '1')
            self.PlotIntervalEntry.grid(
                row=SETTING_ROW_START + 3, column=1, sticky=W)

            self.PlotWindowLabel = Label(self, text='Plot_Window_Second')
            self.PlotWindowLabel.grid(
                row=SETTING_ROW_START + 3, column=2, sticky=W)
            self.PlotWindowEntry = Entry(self, width=20)
            self.PlotWindowEntry.insert(INSERT, '10')
            self.PlotWindowEntry.grid(
                row=SETTING_ROW_START + 3, column=3, sticky=W)

            self.plot_fig = BooleanVar()
            self.PlotFigureLabel = Label(self, text='Plot Figure')
            self.PlotFigureLabel.grid(
                row=SETTING_ROW_START + 4, column=0, sticky=W)
            self.PlotFigureYesRadioButton = Radiobutton(
                self, text='Yes', variable=self.plot_fig, value=True, command=self.__set_widget_enable)
            self.PlotFigureYesRadioButton.grid(
                row=SETTING_ROW_START + 4, column=1, sticky=W)
            self.PlotFigureYesRadioButton.select()
            self.PlotFigureNoRadioButton = Radiobutton(
                self, text='No', variable=self.plot_fig, value=False, command=self.__set_widget_enable)
            self.PlotFigureNoRadioButton.grid(
                row=SETTING_ROW_START + 4, column=2, sticky=W)

            self.pcb_version = IntVar()
            self.PCBVersionLabel = Label(self, text='PCB Version')
            self.PCBVersionLabel.grid(
                row=SETTING_ROW_START + 4, column=3, sticky=W)
            self.PCBVersion11RadioButton = Radiobutton(self, text='1.1', variable=self.pcb_version,
                                                       value=0)
            self.PCBVersion11RadioButton.grid(
                row=SETTING_ROW_START + 4, column=4, sticky=W)
            self.PCBVersion11RadioButton.select()
            self.PCBVersion29RadioButton = Radiobutton(self, text='2.9', variable=self.pcb_version,
                                                       value=1)
            self.PCBVersion29RadioButton.grid(
                row=SETTING_ROW_START + 4, column=5, sticky=W)

            self.acc_plot = []
            for i in range(3):
                self.acc_plot.append(IntVar())
            self.AccLabel = Label(self, text='Acc_Figure')
            self.AccLabel.grid(row=SETTING_ROW_START + 5, column=0, sticky=W)
            self.AccXCheckButton = Checkbutton(
                self, text='X_axis', variable=self.acc_plot[0])
            self.AccXCheckButton.select()
            self.AccXCheckButton.grid(
                row=SETTING_ROW_START + 5, column=1, sticky=W)
            self.AccYCheckButton = Checkbutton(
                self, text='y_axis', variable=self.acc_plot[1])
            self.AccYCheckButton.grid(
                row=SETTING_ROW_START + 5, column=2, sticky=W)
            self.AccZCheckButton = Checkbutton(
                self, text='z_axis', variable=self.acc_plot[2])
            self.AccZCheckButton.grid(
                row=SETTING_ROW_START + 5, column=3, sticky=W)

            self.SaveAsLabel = Label(self, text='Save as')
            self.SaveAsLabel.grid(
                row=SETTING_ROW_START + 6, column=0, sticky=W)
            self.SaveAsButton = Button(
                self, text='...', command=self.__save_as)
            self.SaveAsButton.grid(
                row=SETTING_ROW_START + 6, column=1, sticky=W)
            self.SaveAsFilenameLabel = Label(self, text='')
            self.SaveAsFilenameLabel.grid(
                row=SETTING_ROW_START + 6, column=2, columnspan=2, sticky=W)

            self.fft_enable = IntVar()
            self.FrequencyLabel = Label(
                self, text='Figure_Option', state=DISABLED)
            self.FrequencyLabel.grid(
                row=SETTING_ROW_START + 7, column=0, sticky=W)
            self.FrequencyCheckButton = Checkbutton(
                self, text='FFT', variable=self.fft_enable, state=DISABLED)
            self.FrequencyCheckButton.grid(
                row=SETTING_ROW_START + 7, column=1, sticky=W)
            self.show_both_pressure = BooleanVar()
            self.ShowBothPressureCheckButton = Checkbutton(
                self, text='Show_Both_Pressure', variable=self.show_both_pressure)
            self.ShowBothPressureCheckButton.grid(
                row=SETTING_ROW_START + 7, column=2, sticky=W)

            self.okButton = Button(
                self, text='OK', command=self.__on_ok)
            self.okButton.grid(row=RUN_ROW_START, column=1, sticky=W)
            self.cancelButton = Button(
                self, text='Cancel', command=self.__exit_app)
            self.cancelButton.grid(
                row=RUN_ROW_START, column=2, sticky=W)
            self.stopLogButton = Button(
                self, text='Stop Log', command=self.__stop_log, state=DISABLED)
            self.stopLogButton.grid(
                row=RUN_ROW_START, column=3, sticky=W)
            if ENABLE_STATE_CHANGE == True:
                self.flagButton = Button(
                    self, text='Flag', command=self.__add_flag, state=DISABLED)
                self.flagButton.grid(row=RUN_ROW_START, column=4, sticky=W)

            self.real_duration = IntVar()
            self.RealDurationLabel = Label(self, text='Real Duration')
            self.RealDurationLabel.grid(
                row=STATUS_ROW_START, column=0, sticky=W)
            self.RealDurationValueLabel = Label(
                self, textvariable=self.real_duration)
            self.RealDurationValueLabel.grid(
                row=STATUS_ROW_START, column=1, sticky=W)
            self.RealDurationUnitLabel = Label(self, text='Seconds')
            self.RealDurationUnitLabel.grid(
                row=STATUS_ROW_START, column=2, sticky=W)
            self.ECGStatus = Label(self, text='ECG: Off')
            self.ECGStatus.grid(row=STATUS_ROW_START, column=3, sticky=W)
            if ENABLE_ECG == True:
                self.ECGDeviceLabel = Label(self, text='ECG Device')
                self.ECGDeviceLabel.grid(row=ECG_ROW_START, column=0, sticky=W)
                self.ECGConnectButton = Button(
                    self, text='Connect', command=self.__connect_ECG)
                self.ECGConnectButton.grid(
                    row=ECG_ROW_START, column=1, sticky=W)
                self.ECGStartButton = Button(
                    self, text='Start ECG', command=self.__start_ECG)
                self.ECGStartButton.grid(row=ECG_ROW_START, column=2, sticky=W)
                self.ECGDisconnectButton = Button(
                    self, text='Disconnect', command=self.__disconnect_ECG, state=DISABLED)
                self.ECGDisconnectButton.grid(
                    row=ECG_ROW_START, column=3, sticky=W)
                self.__connect_ECG()
        elif VERSION == Version.DUMMY:
            self.COMLabel = Label(self, text='埠號')
            self.COMLabel.grid(row=SETTING_ROW_START + 1, column=0, sticky=W)
            self.COMCombobox = Combobox(
                self, postcommand=self.__update_com_port)
            self.COMCombobox['values'] = self.com_port
            self.COMCombobox.grid(
                row=SETTING_ROW_START + 1, column=1, sticky=W)
            if len(self.com_port) >= 1:
                self.COMCombobox.current(0)

            self.NameLabel = Label(self, text='姓名')
            self.NameLabel.grid(row=SETTING_ROW_START + 1, column=2, sticky=W)
            self.NameEntry = Entry(self, width=20)
            self.NameEntry.insert(INSERT, 'CHENLONG')
            self.NameEntry.grid(row=SETTING_ROW_START + 1, column=3, sticky=W)

            self.gender_value = BooleanVar()
            self.GenderLabel = Label(self, text='性別')
            self.GenderLabel.grid(
                row=SETTING_ROW_START + 2, column=0, sticky=W)
            self.GenderFemaleRadioButton = Radiobutton(
                self, text='女', variable=self.gender_value, value=True)
            self.GenderFemaleRadioButton.grid(
                row=SETTING_ROW_START + 2, column=1, sticky=W)
            self.GenderMaleRadioButton = Radiobutton(
                self, text='男', variable=self.gender_value, value=False)
            self.GenderMaleRadioButton.grid(
                row=SETTING_ROW_START + 2, column=2, sticky=W)
            self.gender_value.set(True)

            self.WeightLabel = Label(self, text='體重(公斤)')
            self.WeightLabel.grid(
                row=SETTING_ROW_START + 3, column=0, sticky=W)
            self.WeightCombobox = Combobox(self)
            self.WeightCombobox['values'] = WEIGHT_RANGE
            self.WeightCombobox.grid(
                row=SETTING_ROW_START + 3, column=1, sticky=W)
            self.WeightCombobox.current(0)

            self.HeightLabel = Label(self, text='身高(公分)')
            self.HeightLabel.grid(
                row=SETTING_ROW_START + 3, column=2, sticky=W)
            self.HeightCombobox = Combobox(self)
            self.HeightCombobox['values'] = HEIGHT_RANGE
            self.HeightCombobox.grid(
                row=SETTING_ROW_START + 3, column=3, sticky=W)
            self.HeightCombobox.current(0)

            #self.HeightEntry = Entry(self, width=10)
            # self.HeightEntry.grid(
            #    row=SETTING_ROW_START + 3, column=3, sticky=W)

            self.heart_rate_interval_value = IntVar()
            self.HeartRateIntervalLabel = Label(self, text='心律區間')
            self.HeartRateIntervalLabel.grid(
                row=SETTING_ROW_START + 4, column=0, sticky=W)
            self.HeartRateLowIntervalRadioButton = Radiobutton(
                self, text='<= 60 BPM', variable=self.heart_rate_interval_value, value=0)
            self.HeartRateLowIntervalRadioButton.grid(
                row=SETTING_ROW_START + 4, column=1, sticky=W)
            self.HeartRateMiddleIntervalRadioButton = Radiobutton(
                self, text='60~80 BPM', variable=self.heart_rate_interval_value, value=1)
            self.HeartRateMiddleIntervalRadioButton.grid(
                row=SETTING_ROW_START + 4, column=2, sticky=W)
            self.HeartRateHighIntervalRadioButton = Radiobutton(
                self, text='>= 80 BPM', variable=self.heart_rate_interval_value, value=2)
            self.HeartRateHighIntervalRadioButton.grid(
                row=SETTING_ROW_START + 4, column=3, sticky=W)

            self.DurationLabel = Label(self, text='錄製長度(秒)')
            self.DurationLabel.grid(
                row=SETTING_ROW_START + 5, column=0, sticky=W)
            self.DurationEntry = Entry(self, width=20)
            self.DurationEntry.insert(INSERT, '600')
            self.DurationEntry.grid(
                row=SETTING_ROW_START + 5, column=1, sticky=W)

            self.location_value = IntVar()
            self.TesterLocationLabel = Label(self, text='測試者座位')
            self.TesterLocationLabel.grid(
                row=SETTING_ROW_START + 5, column=2, sticky=W)
            self.TesterMainLocationRadioButton = Radiobutton(
                self, text='主駕', variable=self.location_value, value=0)
            self.TesterMainLocationRadioButton.grid(
                row=SETTING_ROW_START + 5, column=3, sticky=W)
            self.TesterCoLocationRadioButton = Radiobutton(
                self, text='副駕', variable=self.location_value, value=1)
            self.TesterCoLocationRadioButton.grid(
                row=SETTING_ROW_START + 5, column=4, sticky=W)

            self.SaveAsLabel = Label(self, text='檔名')
            self.SaveAsLabel.grid(
                row=SETTING_ROW_START + 6, column=0, sticky=W)
            self.SaveAsFilenameLabel = Label(self, text='')
            self.SaveAsFilenameLabel.grid(
                row=SETTING_ROW_START + 6, column=1, columnspan=2, sticky=W)
            self.okButton = Button(
                self, text='開始', command=self.__on_ok)
            self.okButton.grid(row=RUN_ROW_START, column=2, sticky=W)
            self.cancelButton = Button(
                self, text='取消', command=self.__exit_app)
            self.cancelButton.grid(
                row=RUN_ROW_START, column=3, sticky=W)
            self.stopLogButton = Button(
                self, text='停止', command=self.__stop_log, state=DISABLED)
            self.stopLogButton.grid(
                row=RUN_ROW_START, column=4, sticky=W)
            if ENABLE_STATE_CHANGE == True:
                self.flagButton = Button(
                    self, text='記號', command=self.__add_flag, state=DISABLED)
                self.flagButton.grid(row=RUN_ROW_START, column=5, sticky=W)

            self.real_duration = IntVar()
            self.RealDurationLabel = Label(self, text='經過時間')
            self.RealDurationLabel.grid(
                row=STATUS_ROW_START, column=0, sticky=W)
            self.RealDurationValueLabel = Label(
                self, textvariable=self.real_duration)
            self.RealDurationValueLabel.grid(
                row=STATUS_ROW_START, column=1, sticky=W)
            self.RealDurationUnitLabel = Label(self, text='秒')
            self.RealDurationUnitLabel.grid(
                row=STATUS_ROW_START, column=2, sticky=W)
            self.ECGStatus = Label(self, text='ECG: 關閉')
            self.ECGStatus.grid(row=STATUS_ROW_START, column=3, sticky=W)

            if ENABLE_ECG == True:
                self.ECGDeviceLabel = Label(self, text='ECG設備')
                self.ECGDeviceLabel.grid(row=ECG_ROW_START, column=0, sticky=W)
                self.ECGConnectButton = Button(
                    self, text='連接', command=self.__connect_ECG)
                self.ECGConnectButton.grid(
                    row=ECG_ROW_START, column=1, sticky=W)
                self.ECGStartButton = Button(
                    self, text='開始測試ECG', command=self.__start_ECG, state=DISABLED)
                self.ECGStartButton.grid(row=ECG_ROW_START, column=2, sticky=W)
                self.ECGDisconnectButton = Button(
                    self, text='斷線', command=self.__disconnect_ECG, state=DISABLED)
                self.ECGDisconnectButton.grid(
                    row=ECG_ROW_START, column=3, sticky=W)
                self.__connect_ECG()

    def __quit(self):
        self.__exit_app()

    def __set_window_location(self):
        self.master.update()
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
#        self.master.geometry('%dx%d' % (screen_width / 2, screen_height / 4))
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        location = '+%d+%d' % (screen_width / 2 - window_width / 2,
                               screen_height / 2 - window_height / 2)
        self.master.geometry(location)

    def __update_com_port(self):
        self.__get_available_com_port()
        self.COMCombobox['values'] = self.com_port
        if VERSION == Version.PRO:
            self.COMCombobox1['values'] = self.com_port

    def __get_available_com_port(self):
        self.com_port = []
        com_port_list = list(serial.tools.list_ports.comports())
        if len(com_port_list) <= 0:
            if VERSION == Version.PRO:
                messagebox.showerror('Error!', 'No available com port!!')
            elif VERSION == Version.DUMMY:
                messagebox.showerror('錯誤!', '沒有可使用的埠號!!')
        else:
            for i in range(len(com_port_list)):
                self.com_port.append(list(com_port_list[i])[0])
            self.com_port.sort()
            self.com_port.insert(0, '')

    def __set_widget_enable(self):
        if self.plot_fig.get() == True:
            if VERSION == Version.PRO:
                self.PlotIntervalLabel.config(state=NORMAL)
                self.PlotIntervalEntry.config(state=NORMAL)
                self.PlotWindowLabel.config(state=NORMAL)
                self.PlotWindowEntry.config(state=NORMAL)
                self.AccLabel.config(state=NORMAL)
                self.AccXCheckButton.config(state=NORMAL)
                self.AccYCheckButton.config(state=NORMAL)
                self.AccZCheckButton.config(state=NORMAL)
        else:
            if VERSION == Version.PRO:
                self.PlotIntervalLabel.config(state=DISABLED)
                self.PlotIntervalEntry.config(state=DISABLED)
                self.PlotWindowLabel.config(state=DISABLED)
                self.PlotWindowEntry.config(state=DISABLED)
                self.AccLabel.config(state=DISABLED)
                self.AccXCheckButton.config(state=DISABLED)
                self.AccYCheckButton.config(state=DISABLED)
                self.AccZCheckButton.config(state=DISABLED)

    def __save_as(self):
        self.output_filename = filedialog.asksaveasfilename()
        split_filename = os.path.splitext(self.output_filename)
        self.output_filename = split_filename[0] + \
            '_' + str(self.acc_location) + split_filename[1]
        self.SaveAsFilenameLabel.config(text=self.output_filename)
        self.ecg_output_filename = os.path.splitext(self.output_filename)[
            0] + '_ECG' + os.path.splitext(self.output_filename)[1]
        if ENABLE_VIDEO_RECORDING == True:
            self.video_output_filename = os.path.splitext(
                self.output_filename)[0] + '.avi'

    def __exit_app(self):
        global save_log_proc
        global plot_log_proc
        global extract_ecg_proc
        global ecg_ex_pipe

        if ENABLE_ECG and self.ecg_conn_enable == True:
            ecg_ex_pipe[0].send('__EXIT_APP')
            extract_ecg_proc.terminate()

        if 'save_log_proc' in globals().keys():
            save_log_proc.terminate()
        if self.is_plot == True and 'plot_log_proc' in globals().keys():
            plot_log_proc.terminate()
        if VERSION == Version.PRO:
            messagebox.showinfo('Close', 'Exit APP!')
        elif VERSION == Version.DUMMY:
            messagebox.showinfo('關閉', '離開應用程式!')
        self.master.destroy()

    def __stop_log(self):
        global save_log_proc
        global plot_log_proc
        global force_stop_pipe

        self.stopLogButton.config(state=DISABLED)
        self.start_log = False

        if ENABLE_STATE_CHANGE == True:
            self.flagButton.config(state=DISABLED)
        force_stop_pipe[0].send(True)
        # if self.is_plot == True:
        #     plot_log_proc.terminate()

        # if ENABLE_ECG == True:
        #    extract_ecg_proc.terminate()

    def __add_flag(self):
        global ecg_flag_notification
        global bcg_flag_notification
        ecg_flag_notification.value = True
        bcg_flag_notification.value = True
        self.flag_clicked = True
        self.flagButton.configure(fg='red')

    def __on_ok(self):
        if ENABLE_ECG == True:
            if self.ecg_conn_enable == False:
                if VERSION == Version.PRO:
                    messagebox.showerror(
                        'Error', 'Need to connect ECG device firstly!')
                elif VERSION == Version.DUMMY:
                    messagebox.showerror('錯誤', '需先連接ECG設備!')
                return
        self.com_port = self.COMCombobox.get()
        self.current_time = datetime.datetime.now()
        if VERSION == Version.PRO:
            self.com_port1 = self.COMCombobox1.get()
            self.baud_rate = int(self.BaudRateCombobox.get())
            self.sampling_rate = int(self.SamplingRateEntry.get())
            self.plot_interval = int(self.PlotIntervalEntry.get())
            self.plot_duration = int(self.PlotWindowEntry.get())
            self.store_duration = int(self.DurationEntry.get())
            self.downsample = int(self.DownsampleCombobox.get())
            self.fs = self.sampling_rate // self.downsample
            self.is_plot = self.plot_fig.get()
            self.acc_plot_value = [self.acc_plot[0].get(
            ), self.acc_plot[1].get(), self.acc_plot[2].get()]
            self.is_show_both_pressure = self.show_both_pressure.get()
        elif VERSION == Version.DUMMY:
            self.name = self.NameEntry.get()
            if self.gender_value.get() == True:
                self.gender = 0
            else:
                self.gender = 1
            self.weight = int(self.WeightCombobox.get())
            self.height = int(self.HeightCombobox.get())
            self.bpm_interval = self.heart_rate_interval_value.get()
            self.store_duration = int(self.DurationEntry.get())

        config = {}
        config['com_port'] = []
        config['com_port'].append(self.com_port)
        if VERSION == Version.PRO:
            if self.com_port1 != '':
                config['com_port'].append(self.com_port1)
            config['baud_rate'] = self.baud_rate
            config['sampling_rate'] = self.sampling_rate
            config['plot_interval'] = self.plot_interval
            config['plot_duration'] = self.plot_duration
            config['store_duration'] = self.store_duration
            config['downsample'] = self.downsample
            config['fs'] = self.fs
            config['pcb_version'] = self.pcb_version.get()
            config['is_plot'] = self.is_plot
            config['acc_plot_value'] = self.acc_plot_value
            config['output_filename'] = self.output_filename
            config['is_show_both_pressure'] = self.is_show_both_pressure
            if ENABLE_ECG == True:
                config['ecg_output_filename'] = self.ecg_output_filename
            if ENABLE_VIDEO_RECORDING == True:
                config['video_output_filename'] = self.video_output_filename
        elif VERSION == Version.DUMMY:
            config['baud_rate'] = 115200
            config['sampling_rate'] = 64
            config['plot_interval'] = 1
            config['plot_duration'] = 10
            config['store_duration'] = self.store_duration
            config['downsample'] = 1
            config['fs'] = config['sampling_rate'] // config['downsample']
            config['pcb_version'] = 0
            config['is_plot'] = True
            config['acc_plot_value'] = [1, 1, 1]
            config['tester_location'] = self.location_value.get()
            self.output_filename = config['output_filename'] = self.name + '_'\
                + str(self.weight) + '_'\
                + str(self.height) + '_'\
                + str(self.gender) + '_'\
                + str(self.bpm_interval) + '_'\
                + str(config['tester_location']) + '_'\
                + str(self.current_time.year)\
                + '%02d' % self.current_time.month\
                + '%02d' % self.current_time.day\
                + '%02d' % self.current_time.hour\
                + '%02d' % self.current_time.minute\
                + '%02d' % self.current_time.second + '_'\
                + str(self.acc_location)\
                + '.log'
            config['is_show_both_pressure'] = False
            self.SaveAsFilenameLabel.config(text=config['output_filename'])
            if ENABLE_ECG == True:
                config['ecg_output_filename'] = os.path.splitext(config['output_filename'])[
                    0] + '_ECG' + os.path.splitext(config['output_filename'])[1]
                self.ecg_output_filename = config['ecg_output_filename']

        if self.com_port == '':
            if VERSION == Version.PRO:
                messagebox.showerror('Error!', 'Choose com port to connect!!')
            elif VERSION == Version.DUMMY:
                messagebox.showerror('錯誤!', '選擇連接com port!!')
            return

        if VERSION == Version.PRO:
            if self.baud_rate == '':
                if VERSION == Version.PRO:
                    messagebox.showerror(
                        'Error!', 'Choose baud rate to connect!!')
                elif VERSION == Version.DUMMY:
                    messagebox.showerror('錯誤!', '選擇連接波特率!!')
                return
            else:
                self.baud_rate = int(self.baud_rate)

            if self.output_filename == '':
                if VERSION == Version.PRO:
                    messagebox.showerror(
                        'Error!', 'Choose filename to save output!!')
                elif VERSION == Version.DUMMY:
                    messagebox.showerror(
                        '錯誤!', '選擇儲存資料檔案名稱!!')
                return

            if os.path.exists(config['output_filename']):
                if not messagebox.askyesno('To overwrite?', 'To overwrite this original file?'):
                    if VERSION == Version.PRO:
                        messagebox.showerror(
                            'Error!', 'To select again other file!')
                    elif VERSION == Version.DUMMY:
                        messagebox.showerror(
                            '錯誤!', '重新選擇一個檔案名稱!!')
                    return

        self.start_log = True
        self.stopLogButton.config(state=NORMAL)
        process_log(config)
        # threading.Thread(target = self.save_log).start()
        # threading.Thread(target = self.plot_log).start()
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.master.after(1000, self.update_duration)
        if ENABLE_STATE_CHANGE == True:
            self.flagButton.config(state=NORMAL)
        # self.timer = threading.Timer(1, self.update_duration)
        # self.timer.daemon = True
        # self.timer.start()
        # plt.ion()

    def update_duration(self):
        global result_pipe
        global ecg_ex_pipe

        global extract_ecg_proc
        if self.flag_clicked == True:
            self.flag_delay_sec += 1
            if self.flag_delay_sec >= 5:
                self.flagButton.configure(fg='black')
                self.flag_clicked = False
                self.flag_delay_sec = 0
        if ENABLE_ECG == True:
            if ecg_ex_pipe[0].poll():
                if ecg_ex_pipe[0].recv() == '__ERR_ECG':
                    self.__stop_log()
                    self.__disconnect_ECG()
                    if VERSION == Version.PRO:
                        messagebox.showerror('Error!', 'ECG extraction fails!')
                    elif VERSION == Version.DUMMY:
                        messagebox.showerror('錯誤!', 'ECG擷取失敗!')
            if extract_ecg_proc.is_alive() == False:
                self.__stop_log()
                self.__disconnect_ECG()
                if VERSION == Version.PRO:
                    messagebox.showerror('Error!', 'ECG extraction fails!')
                elif VERSION == Version.DUMMY:
                    messagebox.showerror('錯誤!', 'ECG擷取失敗!')
        log_duration = int(
            datetime.datetime.now().timestamp() - self.start_timestamp)

        # if ENABLE_ECG == True:
        #     try:
        #         # Read samples
        #         ecg_data = self.ecg_device.read(self.n_samples)
        #         self.ecg_output.writerows(ecg_data.tolist())
        #         self.ecg_fp.flush()
        #     except Exception as e:
        #         print('Extract ECG error:', str(e))

        if not save_log_proc.is_alive():
            self.start_log = False
            app_win.stopLogButton.config(state=DISABLED)

            if VERSION == Version.DUMMY:
                if not os.path.exists(self.output_filename) or os.path.getsize(self.output_filename) == 0:
                    messagebox.showerror('錯誤!', '儲存BCG檔案錯誤!!')
                    return
                if ENABLE_ECG == True:
                    if not os.path.exists(self.ecg_output_filename) or os.path.getsize(self.ecg_output_filename) == 0:
                        messagebox.showerror('錯誤!', '儲存ECG檔案錯誤!!')
                        return
            messagebox.showinfo('Notification!', 'Finish!')
            return

        if self.start_log == False:
            return
        else:
            self.real_duration.set(
                int(datetime.datetime.now().timestamp() - self.start_timestamp))

        try:
            heart_rate, res_rate, algo_state = result_pipe[0].recv()
        except EOFError:
            result_pipe[0].close()
        self.BpmValue.set(heart_rate)
        self.RpmValue.set(res_rate)
        self.SignalQualityValue.set(algo_state)
        self.master.after(1000, self.update_duration)

    def __find_ECG_mac_addr(self):
        ecg_mac_addrs = []
        all_BT_devices = bitalino.find()
        for BT_device in all_BT_devices:
            if BT_device[1].lower().find('bitalino') == -1:
                continue
            else:
                ecg_mac_addrs.append(BT_device[0])

        for ecg_mac_addr in ecg_mac_addrs:
            self.ecg_mac_addr = ecg_mac_addr
            try:
                device = BITalino(self.ecg_mac_addr, timeout=ECG_CONN_TIMEOUT)
                self.ecg_mac_addr_found = True
                device.close()
                break
            except OSError as e:
                self.ecg_mac_addr_found = False

    def __connect_ECG(self):
        global extract_ecg_proc
        global ecg_connect_notification_pipe
        global ecg_ex_pipe
        global log_done
        global ecg_flag_notification
        if self.ecg_mac_addr_found == True:
            print('Find mac_addr!!')
            log_done = multiprocessing.Value('i', False)
            ecg_connect_notification_pipe = multiprocessing.Pipe()
            ecg_ex_pipe = multiprocessing.Pipe()
            extract_ecg_proc = multiprocessing.Process(target=extract_ecg, args=(
                self.ecg_mac_addr, ecg_connect_notification_pipe[0], ecg_ex_pipe[1], ecg_flag_notification, log_done))
            extract_ecg_proc.start()
            self.ecg_conn_enable = ecg_connect_notification_pipe[1].recv()
            if self.ecg_conn_enable == True:
                self.ECGDisconnectButton.config(state=NORMAL)
                self.ECGConnectButton.config(state=DISABLED)
                if VERSION == Version.PRO:
                    self.ECGStatus['text'] = 'ECG: On'
                    messagebox.showinfo('Notification!', 'ECG Connected!')
                elif VERSION == Version.DUMMY:
                    self.ECGStatus['text'] = 'ECG: 開啟'
                    messagebox.showinfo('通知!', 'ECG連接!')
            else:
                self.ECGDisconnectButton.config(state=DISABLED)
                self.ECGConnectButton.config(state=NORMAL)
                if VERSION == Version.PRO:
                    self.ECGStatus['text'] = 'ECG: Off'
                    messagebox.showerror('Error!', 'ECG Not Connected!')
                elif VERSION == Version.DUMMY:
                    self.ECGStatus['text'] = 'ECG: 關閉'
                    messagebox.showerror('錯誤!', 'ECG未連接!')
        else:
            if VERSION == Version.PRO:
                messagebox.showerror('Error!', 'ECG Not Connected!')
            elif VERSION == Version.DUMMY:
                messagebox.showerror('錯誤!', 'ECG未連接!')

    def __disconnect_ECG(self):
        self.ecg_conn_enable == False
        ecg_ex_pipe[0].send('__EXIT_APP')
        if VERSION == Version.PRO:
            self.ECGStatus['text'] = 'ECG: Off'
        elif VERSION == Version.DUMMY:
            self.ECGStatus['text'] = 'ECG: 關閉'
        self.ECGDisconnectButton.config(state=DISABLED)
        self.ECGConnectButton.config(state=NORMAL)

    def __start_ECG(self):
        pass


def save_log(send_pipe, result_pipe_in, force_stop_pipe_in, config, bcg_flag_notification, log_done):
    time_counter = 0
    acc = [0, 0, 0]
    # fft_pressure = []
    # fft_acc = []
    # is_first_filter = True
    # first_filter = 0
    header_flag = False

    sers = []
    try:
        for serial_port in config['com_port']:
            sers.append(serial.Serial(serial_port, config['baud_rate'],
                                      BYTESIZE, PARITY, STOPBITS))

        with open(config['output_filename'], 'w') as output_stream:
            current_start_time = datetime.datetime.now()
            output_stream.write('Start Time: ' + str(current_start_time.year) + '/' + str(current_start_time.month) + '/' + str(
                current_start_time.day) + '-' + str(current_start_time.hour) + ':' + str(current_start_time.minute) + ':' + str(current_start_time.second) + ':' + str(current_start_time.microsecond) + '\n')
            while True:
                timestamp = []
                pressure = []
                heart_rate = []
                res_rate = []
                algo_state = []
                for i in range(3):
                    acc[i] = []
                for i, ser in enumerate(sers):
                    while header_flag != True:
                        a_byte = int.from_bytes(
                            ser.read(1), byteorder='little')
                        if a_byte == HEADER_TRAILER[0]:
                            a_byte = int.from_bytes(
                                ser.read(1), byteorder='little')
                            if a_byte == HEADER_TRAILER[1]:
                                header_flag = True
                    if i == 0:
                        time_counter += 1

                    if time_counter % config['downsample'] == 0:
                        if config['pcb_version'] == 0:
                            pressure.append(int.from_bytes(
                                ser.read(3), byteorder='little', signed=True))
                        elif config['pcb_version'] == 1:
                            pressure.append(int.from_bytes(
                                ser.read(3), byteorder='little', signed=False))
                        heart_rate.append(int.from_bytes(
                            ser.read(1), byteorder='little', signed=True))
                        res_rate.append(int.from_bytes(
                            ser.read(1), byteorder='little', signed=True))
                        system_state = int.from_bytes(
                            ser.read(1), byteorder='little')
                        curve_state = system_state & 0xF
                        algo_state.append((system_state & 0xF0) >> 4)

                        if config['is_show_both_pressure'] == True:
                            pressure2 = int.from_bytes(
                                ser.read(3), byteorder='little', signed=False)
                            _ = int.from_bytes(
                                ser.read(3), byteorder='little', signed=False)
                        else:
                            acc[0].append(int.from_bytes(
                                ser.read(2), byteorder='little', signed=True))
                            acc[1].append(int.from_bytes(
                                ser.read(2), byteorder='little', signed=True))
                            acc[2].append(int.from_bytes(
                                ser.read(2), byteorder='little', signed=True))
                        timestamp.append(int.from_bytes(
                            ser.read(3), byteorder='little'))
                        trailer = int.from_bytes(
                            ser.read(1), byteorder='little')
                        header_flag = False
                    else:
                        ser.read(16)
                        header_flag = False
                        continue

                    if trailer != HEADER_TRAILER[2]:
                        continue
                # if is_first_filter == True:
                #    first_filter = pressure
                #    is_first_filter = False
                if config['is_show_both_pressure'] == True:
                    output_stream.write(str(timestamp) + ', ' + str(pressure) + ', '
                                        + str(pressure2) + ', ' +
                                        str(heart_rate) + ','
                                        + str(res_rate) + ',' + str(algo_state) + '\n')
                else:
                    if len(pressure) == 2:
                        output_stream.write(str(timestamp[0]) + ', ' + str(pressure[0]) + ', '
                                            + str(acc[0][0]) + ', ' +
                                            str(acc[1][0]) + ', '
                                            + str(acc[2][0]) + ',' +
                                            str(heart_rate[0]) + ','
                                            + str(res_rate[0]) + ',' +
                                            str(algo_state[0]) + ': '
                                            + str(timestamp[1]) + ', ' +
                                            str(pressure[1]) + ', '
                                            + str(acc[0][1]) + ', ' +
                                            str(acc[1][1]) + ', '
                                            + str(acc[2][1]) + ',' +
                                            str(heart_rate[1]) + ','
                                            + str(res_rate[1]) + ',' + str(algo_state[1]) + '\n')
                    else:
                        output_stream.write(str(timestamp[0]) + ', ' + str(pressure[0]) + ', '
                                            + str(acc[0][0]) + ', ' +
                                            str(acc[1][0]) + ', '
                                            + str(acc[2][0]) + ',' +
                                            str(heart_rate[0]) + ','
                                            + str(res_rate[0]) + ',' + str(algo_state[0]) + '\n')
                output_stream.flush()

                if bcg_flag_notification.value == True:
                    output_stream.write('State Change\n')
                    bcg_flag_notification.value = False

                if config['is_plot'] == True:
                    if config['is_show_both_pressure'] == True:
                        send_pipe.send([round(time_counter / config['sampling_rate'], 3),
                                        pressure, pressure2])
                    else:
                        if len(pressure) != 2:
                            send_pipe.send([round(time_counter / config['sampling_rate'], 3),
                                            pressure, [acc[0], acc[1], acc[2]]])
                        else:
                            send_pipe.send([round(time_counter / config['sampling_rate'], 3),
                                            pressure[0], pressure[1]])
                # fft_pressure.append(float(pressure))
                # fft_acc.append(acc[1])
                is_force_stop = False
                if force_stop_pipe_in.poll():
                    is_force_stop = force_stop_pipe_in.recv()

                if time_counter >= config['store_duration'] * config['sampling_rate'] or is_force_stop:
                    log_done.value = True
                    result_pipe_in.send((heart_rate, res_rate, algo_state))
                    result_pipe_in.close()
                    force_stop_pipe_in.close()
                    current_start_time = datetime.datetime.now()
                    print('End save log!!')
                    output_stream.write('End Time: ' + str(current_start_time.year) + '/' + str(current_start_time.month) + '/' + str(
                        current_start_time.day) + '-' + str(current_start_time.hour) + ':' + str(current_start_time.minute) + ':' + str(current_start_time.second) + ':' + str(current_start_time.microsecond) + '\n')
                    print('log_done.value = True')
                    break
                if time_counter % config['sampling_rate'] == 0:
                    result_pipe_in.send((heart_rate, res_rate, algo_state))
    except serial.SerialException:
        app_win.stopLogButton.config(state=DISABLED)
        print('Serial Port Error!')
    else:
        ser.close()


def plot_log(recv_pipe, config, log_done):
    global app_win
    temp_time_x = []
    temp_pressure_y = []
    temp_acc_y = []
    recv_counter = 0

    while True:
        try:
            if log_done.value == False and config['is_plot'] == True:
                temp = recv_pipe.recv()
                recv_counter += 1
                temp_time_x.append(temp[0])
                temp_pressure_y.append(temp[1])
                temp_acc_y.append(temp[2])
                if recv_counter > (config['sampling_rate'] / config['downsample']) * config['plot_duration']:
                    temp_time_x.pop(0)
                    temp_pressure_y.pop(0)
                    temp_acc_y.pop(0)
                if recv_counter % (config['plot_interval'] * config['sampling_rate'] / config['downsample']) != 0:
                    continue

                plt.ion()
                plt.figure(1)
                plt.clf()
                if len(config['com_port']) == 2:
                    plt.subplot(411)
                    plt.plot(temp_time_x, temp_pressure_y)
                    plt.xlabel('Time(s)')
                    plt.ylabel('Pressure')
                    plt.subplot(412)
                    plt.plot(temp_time_x, temp_acc_y)
                    plt.xlabel('Time(s)')
                    plt.ylabel('Pressure')
                else:
                    ax = plt.subplot(411)
                    plt.plot(temp_time_x, temp_pressure_y)
                    plt.xlabel('Time(s)')
                    plt.ylabel('Pressure')
                    for i in range(3):
                        if config['acc_plot_value'][i] == 1:
                            plt.subplot(412 + i)
                            if config['is_show_both_pressure'] == True:
                                plt.plot(temp_time_x, temp_acc_y)
                            else:
                                plt.plot(temp_time_x, [item[i]
                                                       for item in temp_acc_y])
                            plt.xlabel('Time(s)')
                            plt.ylabel(ACC_LABEL_Y_AXIS[i])
                # if self.fft_enable.get() == 1 and len(fft_pressure) % self.fs * self.fft_interval == 0:
                #    filtered_fft_pressure = [
                #        (a - first_filter) for a in fft_pressure]
                # plt.ion()
        #               nyq = 0.5 * 60
        #               low = 0.7 / nyq
        #               high = 10 / nyq
        #               b, a = butter(4, [low, high], btype='band')
        #
        #               filtered_fft_pressure = lfilter(b, a, fft_pressure)
        #               data1 = np.array(filtered_fft_pressure)
        #               data2 = np.array(fft_pressure)
                # data = np.array(fft_pressure)
        #               p_s = librosa.core.stft(data1, n_fft = self.fs*20)  # Short-time Fourier transform
        #               p_ss = np.array(np.abs(p_s))  # get magnitude
        #               p_ss = np.sqrt(p_ss)
        #               fft_acc = [float(a) for a in fft_acc]
        #               acc_s = librosa.core.stft(data2, n_fft = self.fs*20)  # Short-time Fourier transform
        #               acc_ss = np.array(np.abs(acc_s))  # get magnitude
                # fft_acc = [float(a) for a in fft_acc]
                # data = np.array(fft_acc)
                # acc_s = librosa.core.stft(data, n_fft = self.fs*20)  # Short-time Fourier transform
                # acc_ss = np.array(np.abs(acc_s))  # get magnitude
        #               for i in range(14):
        #    p_ss[i] = 0
        #                   acc_ss[i] = 0
                        # fft_pressure = []
                        # fft_acc = []
        #               plt.figure(2)
        # plt.clf()
        #               plt.subplot(211)
        #               plt.pcolormesh(p_ss)
        #               plt.grid(axis = 'y')
        #               plt.ylim([0, 150])
            # axs[0].set_title('p-sensor')
        #               plt.subplot(212)
        #               plt.pcolormesh(acc_ss)
        #               plt.grid(axis = 'y')
        #               plt.ylim([0, 150])
                    # plt.pcolormesh(acc_ss)
                    # axs[1].set_title('p-sensor')

                    # fig, axs = plt.subplots(2, 1)
                    # axs[0].pcolormesh(p_ss)
                    # axs[0].set_title('p-sensor')
                    # axs[1].pcolormesh(acc_ss)
                    # axs[1].set_title('p-sensor')

                plt.pause(0.001)
                plt.ioff()
            else:
                print('Plot Done!!')
                break
        except Exception as e:
            print('Plot error:', str(e))
    print('Plot process finish!!')


def record_video(config, log_done):
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = None
    (h, w) = (None, None)
    zeros = None

    time.sleep(1.0)
    start = time.time()
    last_capture_time = 0
    while True:
        this_capture_time = time.time()
        if last_capture_time == 0 or (this_capture_time - last_capture_time) >= 1/25:
            last_capture_time = this_capture_time
            frame = vs.read()
            current_time = datetime.datetime.now()
            frame = imutils.resize(frame, width=240)
            cv2.putText(frame, str(current_time), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if writer is None:
                (h, w) = frame.shape[:2]
                writer = cv2.VideoWriter(
                    config['video_output_filename'], fourcc, 25, (w, h), True)

            writer.write(frame)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                ear = round(ear, 3)
                # LIST_EAR.append(str(ear))

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)

        if log_done.value == True:
            end = time.time()
            print('total time' + str(end-start))
            cv2.destroyAllWindows()
            writer.release()
            vs.stop()
            break


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


def extract_ecg(ecg_mac_addr, conn_pipe, ex_pipe, ecg_flag_notification, log_done):
    battery_threshold = 30
    acq_channels = [0, 1]
    sampling_rate = 100
    n_samples = 100
    try:
        ecg_device = BITalino(ecg_mac_addr, timeout=ECG_CONN_TIMEOUT)
        conn_pipe.send(True)
    except OSError as e:
        conn_pipe.send(False)
        return

    ecg_device.battery(battery_threshold)
    print(ecg_device.version())

    while True:
        ecg_output_filename = ex_pipe.recv()
        if ecg_output_filename == '__EXIT_APP':
            ecg_device.close()
            break
        ecg_device.start(sampling_rate, acq_channels)
        with open(ecg_output_filename, 'w', encoding='utf-8', newline='') as ecg_output:
            current_time = datetime.datetime.now()
            ecg_output.write('Start Time: '
                             + str(current_time.year)
                             + '/' + str(current_time.month)
                             + '/' + str(current_time.day)
                             + '-' + str(current_time.hour)
                             + ':' + str(current_time.minute)
                             + ':' + str(current_time.second)
                             + ':' + str(current_time.microsecond)
                             + '\n')
            ecg_csv_writer = csv.writer(ecg_output, quoting=csv.QUOTE_NONE)
            while True:
                if log_done.value == False:
                    # Read samples
                    try:
                        ecg_data = ecg_device.read(n_samples)
                        ecg_csv_writer.writerows(ecg_data.tolist())
                        ecg_output.flush()
                    except Exception as e:
                        print('Extract ECG error:', str(e))
                        ex_pipe.send('__ERR_ECG')
                        break
                else:
                    ecg_device.stop()
                    current_time = datetime.datetime.now()
                    ecg_output.write('End Time: '
                                     + str(current_time.year)
                                     + '/' + str(current_time.month)
                                     + '/' + str(current_time.day)
                                     + '-' + str(current_time.hour)
                                     + ':' + str(current_time.minute)
                                     + ':' + str(current_time.second)
                                     + ':' + str(current_time.microsecond)
                                     + '\n')
                    print('Extract ECG Done!!')
                    break

                if ecg_flag_notification.value == True:
                    ecg_output.write('State Change\n')
                    ecg_flag_notification.value = False

                if ex_pipe.poll():
                    if ex_pipe.recv() == '__EXIT_APP':
                        ecg_device.close()
                        return
    print('Extract ECG process finish!!')


def process_log(config):
    global log_done
    # global queue
    global record_video_proc
    global save_log_proc
    global plot_log_proc
    global extract_ecg_proc
    global result_pipe
    global sensor_data_pipe
    global force_stop_pipe
    global ecg_connect_notification_pipe
    global ecg_ex_pipe
    global bcg_flag_notification

    result_pipe = multiprocessing.Pipe()
    sensor_data_pipe = multiprocessing.Pipe()
    force_stop_pipe = multiprocessing.Pipe()
    log_done.value = False

    if ENABLE_VIDEO_RECORDING == True:
        record_video_proc = multiprocessing.Process(
            target=record_video, args=(config, log_done))
        record_video_proc.start()
    if ENABLE_ECG == True:
        ecg_ex_pipe[0].send(config['ecg_output_filename'])

    save_log_proc = multiprocessing.Process(target=save_log, args=(
        sensor_data_pipe[0], result_pipe[1], force_stop_pipe[1], config, bcg_flag_notification, log_done))
    save_log_proc.start()

    if config['is_plot'] == True:
        plot_log_proc = multiprocessing.Process(
            target=plot_log, args=(sensor_data_pipe[1], config, log_done))
        plot_log_proc.start()


if __name__ == '__main__':
    VERSION = Version.PRO
    #VERSION = Version.DUMMY
    app_win = AppFigurePlot()
    app_win.mainloop()
