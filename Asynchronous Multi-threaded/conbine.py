import subprocess
import time
import serial
import threading
import chardet  # 自动检测编码

# 串口配置
SERIAL_PORT = '/dev/ttyS1'  # 根据实际的串口端口调整
BAUD_RATE = 115200
TIMEOUT = 0.1  # 超时时间

process = None  # 进程管理
flag = 0  # 标记进程是否正在运行


def start_b_py():
    """启动 main.py 进程"""
    global process
    command = ['python', '/home/orangepi/Desktop/704/serial/1919example/main.py']

    # 以非阻塞模式运行，并处理 stdout/stderr
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # 行缓冲，防止 stdout 卡住
        universal_newlines=False  # 关闭自动解码，手动处理编码
    )

    # 启动线程定期读取 stdout 和 stderr
    threading.Thread(target=read_output, args=(process.stdout, "STDOUT"), daemon=True).start()
    threading.Thread(target=read_output, args=(process.stderr, "STDERR"), daemon=True).start()

    return process


def read_output(pipe, pipe_name):
    """非阻塞方式持续读取 stdout/stderr，防止缓冲区溢出"""
    try:
        raw_data = pipe.read()  # 读取所有数据
        if not raw_data:
            return

        # 自动检测编码
        detected_encoding = chardet.detect(raw_data)["encoding"]
        encoding = detected_encoding if detected_encoding else "utf-8"

        # 解码数据，并忽略错误
        decoded_data = raw_data.decode(encoding, errors="ignore")

        for line in decoded_data.splitlines():
            print(f"[{pipe_name}] {line.strip()}")
    except Exception as e:
        print(f"[{pipe_name} 读取错误] {e}")


def stop_b_py():
    """停止 main.py 进程"""
    global process
    if process:
        try:
            print("正在停止 main.py...")
            process.terminate()  # 发送终止信号
            process.wait(timeout=5)  # 等待进程退出
            print("main.py 进程已终止")
        except subprocess.TimeoutExpired:
            print("main.py 进程未能正常终止，尝试强制杀死...")
            process.kill()  # 强制终止
            process.wait()
            print("main.py 进程已被强制终止")
        finally:
            process = None  # 释放进程对象
    else:
        print("没有正在运行的 main.py 进程")


def handle_serial_input(ser):
    """读取串口数据并控制 main.py 启动或停止"""
    global process, flag
    while True:
        if ser.in_waiting > 0:
            try:
                data = ser.read().decode('utf-8', errors="ignore").strip()
                print(f"接收到串口数据: {data}")

                if data == 'T' and flag == 0:  # 启动进程
                    print("启动 main.py...")
                    process = start_b_py()
                    flag = 1

                elif data == 'T' and flag == 1:  # 启动进程
                    ser.write("POSReady\r\n".encode())

                elif data == 'F':  # 终止进程
                    print("停止 main.py 进程...")
                    stop_b_py()
                    ser.write("POSEnd\r\n".encode())
                    flag = 0

                elif data == 'Q':  # 退出程序
                    print("退出程序...")
                    return
                else:
                    print(f"无效的串口命令: {data}")
            except Exception as e:
                print(f"串口读取错误: {e}")

        time.sleep(0.1)


def check_process():
    """定期检查 main.py 是否运行中，防止卡死"""
    global process, flag
    while True:
        if flag == 1 and (process is None or process.poll() is not None):  # 进程已退出
            print("检测到 main.py 进程已退出，尝试重启...")
            process = start_b_py()
        time.sleep(5)  # 每 5 秒检查一次


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        if not ser.is_open:
            ser.open()
        print("串口已打开，等待数据接收...")

        # 启动串口输入处理线程
        serial_thread = threading.Thread(target=handle_serial_input, args=(ser,))
        serial_thread.daemon = True
        serial_thread.start()

        # 启动进程监控线程，防止 main.py 进程意外退出
        monitor_thread = threading.Thread(target=check_process, daemon=True)
        monitor_thread.start()

        serial_thread.join()
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("串口已关闭。")


if __name__ == "__main__":
    main()
