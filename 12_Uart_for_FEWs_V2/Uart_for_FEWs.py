import serial
import time
from base.packet_python import MAVLink
from datetime import datetime
f = open('raw_data_nrf.txt', 'wb')
mav = MAVLink(f)


def generate_filename_with_timestamp():
    # 獲取當下日期和時間
    now = datetime.now()
    # 格式化日期和時間，並用作檔名
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp


def send_at_command(ser, at_cmd, add_crlf=True, timeout=1, encoding='utf-8', errors='ignore'):
    if add_crlf:
        ser.write((at_cmd + '\r\n').encode(encoding))
    else:
        ser.write(at_cmd.encode(encoding))
    time.sleep(timeout)
    response = ser.read(ser.in_waiting).decode(encoding, errors=errors)
    return response.strip()


def decode_by_mavlink(ser, recv=False):
    raw_show_downsample = 10
    count = 0
    
        # 生成檔名
    filename = generate_filename_with_timestamp() + ".log"
    # 生成檔案
    with open(filename, 'w') as file:
        while True:
            if recv == False:
                break
            if recv:
                try:
                    raw_data = ser.read(30)
                    # print(raw_data)
                    msgs = mav.parse_buffer(raw_data)
                    if msgs:
                        for msg in msgs:
                            # Raw data
                            if msg.id == 22:
                                count += 1
                                line = str(msg.timestamp) + ',' + str(msg.pressure) + ',' + str(msg.accx) + ',' + str(msg.accy) + ',' + str(msg.accz)
                                file.write(line + '\n')
                                if count == raw_show_downsample:
                                    print(msg)
                                    count = 0
                            # algo data
                            if msg.id == 23:
                                print(msg)
                            
                            # fatigue data
                            if msg.id == 24:
                                print(msg)
                except Exception as e:
                    print(f"Error while decoding MAVLink data: {e}")


def main():
    # 串口设置
    port = 'COM3'  # 修改为你的串口号，如'/dev/ttyUSB0'（Linux）或'COM1'（Windows）
    baud_rate = 115200  # 波特率，通常为115200
    timeout = 1  # 读取串口数据的超时时间（秒）

    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        if ser.is_open:
            print(f"open {port}")

            try:
                while True:
                    at_cmd = 'run mode'
                    # 从用户输入获取命令
                    # at_cmd = input("key command (key'exit'to quit): ").strip()
                    # if at_cmd.lower() == 'exit':
                    #     break

                    # # 检查是否需要添加回车换行符
                    # # add_crlf = True
                    # # 根据设备文档判断是否需要添加回车换行符
                    # if at_cmd == '+++':
                    #     add_crlf = False
                    #     # 发送AT命令并读取响应
                    #     response = send_at_command(ser, at_cmd, add_crlf)
                    #     print(f"key command: {at_cmd}\n respond: {response}")
                    # elif at_cmd == 'AT+EXIT':
                    add_crlf = True
                    response = send_at_command(ser, at_cmd, add_crlf)
                    # print(f"key command: {at_cmd}\n respond: {response}")
                    # do_what = input('use decode with mavlink?(1:yes  ,2:no)')
                    # if int(do_what) == 1:
                    decode_by_mavlink(ser, recv=True)
                    # else:
                    #     pass
                    # else:
                    #     add_crlf = True
                    #     response = send_at_command(ser, at_cmd, add_crlf)
                    #     print(f"key command: {at_cmd}\n respond: {response}")
            except KeyboardInterrupt:
                print("Stop.")

            # 关闭串口
            ser.close()
            print("close port")
        else:
            print(f"can not open the port {port}")
    except serial.SerialException as e:
        print(f"port error: {e}")


if __name__ == "__main__":
    main()
