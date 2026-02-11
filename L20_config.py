#!/usr/bin/env python3
import time
from typing import List
from pymodbus.client import ModbusSerialClient

_INTERVAL = 0.005


class LinkerHandL20RS485:
    """
    L20 - 20自由度版本
    """

    KEYS = [
        # pitch
        "thumb_pitch", "index_pitch", "middle_pitch",
        "ring_pitch", "little_pitch",
        # yaw
        "thumb_yaw", "index_yaw", "middle_yaw",
        "ring_yaw", "little_yaw",
        # roll
        "thumb_roll", "index_roll", "middle_roll",
        "ring_roll", "little_roll",
        # tip
        "thumb_tip", "index_tip", "middle_tip",
        "ring_tip", "little_tip",
    ]

    def __init__(self, hand_id=0x27, modbus_port="/dev/ttyUSB0", baudrate=115200):
        self.slave = hand_id
        self.cli = ModbusSerialClient(
            port=modbus_port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.05,
            retries=3,
            retry_on_empty=True,
            handle_local_echo=False
        )

        self.connected = self.cli.connect()
        if not self.connected:
            raise ConnectionError(f"RS485 connect fail to {modbus_port}")

    # --------------------------------------------------
    # 读取接口
    # --------------------------------------------------

    def read_angles(self) -> List[int]:
        """读取 20 个关节角度 (输入寄存器 0-19)"""
        time.sleep(_INTERVAL)
        rsp = self.cli.read_input_registers(address=0, count=20, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"read_angles failed: {rsp}")
        return rsp.registers

    def read_speeds(self) -> List[int]:
        """读取 5 个实际速度 (20-24)"""
        time.sleep(_INTERVAL)
        rsp = self.cli.read_input_registers(address=20, count=5, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"read_speeds failed: {rsp}")
        return rsp.registers

    def read_currents(self) -> List[int]:
        """读取 5 个电流 (25-29)"""
        time.sleep(_INTERVAL)
        rsp = self.cli.read_input_registers(address=25, count=5, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"read_currents failed: {rsp}")
        return rsp.registers

    def read_faults(self) -> List[int]:
        """读取 5 个故障码 (30-34)"""
        time.sleep(_INTERVAL)
        rsp = self.cli.read_input_registers(address=30, count=5, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"read_faults failed: {rsp}")
        return rsp.registers

    def read_normal_force(self) -> List[int]:
        """读取 5 个法向力 (35-39)"""
        time.sleep(_INTERVAL)
        rsp = self.cli.read_input_registers(address=35, count=5, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"read_normal_force failed: {rsp}")
        return rsp.registers

    # --------------------------------------------------
    # 写入接口 (保持寄存器)
    # --------------------------------------------------

    def write_angles(self, vals: List[int]):
        """写入 20 个关节角度 (保持寄存器 0-19)"""
        if len(vals) != 20:
            raise ValueError("需要 20 个 0-255 整数")

        vals = [int(x) for x in vals]
        time.sleep(_INTERVAL)
        rsp = self.cli.write_registers(address=0, values=vals, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"write_angles failed: {rsp}")

    def write_speeds(self, vals: List[int]):
        """写入 5 个最大速度 (20-24)"""
        if len(vals) != 5:
            raise ValueError("需要 5 个 0-255 整数")

        vals = [int(x) for x in vals]
        time.sleep(_INTERVAL)
        rsp = self.cli.write_registers(address=20, values=vals, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"write_speeds failed: {rsp}")

    def write_currents(self, vals: List[int]):
        """写入 5 个过流阈值 (25-29)"""
        if len(vals) != 5:
            raise ValueError("需要 5 个 0-255 整数")

        vals = [int(x) for x in vals]
        time.sleep(_INTERVAL)
        rsp = self.cli.write_registers(address=25, values=vals, slave=self.slave)
        if rsp.isError():
            raise RuntimeError(f"write_currents failed: {rsp}")

    def clear_fault(self, finger_index: int):
        """
        清除单个手指故障 (30-34)
        finger_index: 0-4
        """
        if finger_index < 0 or finger_index > 4:
            raise ValueError("finger_index 必须在 0-4")

        time.sleep(_INTERVAL)
        rsp = self.cli.write_register(
            address=30 + finger_index,
            value=1,
            slave=self.slave
        )
        if rsp.isError():
            raise RuntimeError(f"clear_fault failed: {rsp}")

    # --------------------------------------------------
    # API封装
    # --------------------------------------------------

    def set_joint_positions(self, joint_angles=None):
        joint_angles = joint_angles or [0] * 20
        self.write_angles(joint_angles)

    def set_speed(self, speed=None):
        speed = speed or [200] * 5
        self.write_speeds(speed)

    def set_current(self, current=None):
        current = current or [200] * 5
        self.write_currents(current)

    def get_state(self):
        return self.read_angles()

    def get_speed(self):
        return self.read_speeds()

    def get_current(self):
        return self.read_currents()

    def get_fault(self):
        return self.read_faults()

    def get_normal_force(self):
        return self.read_normal_force()

    # --------------------------------------------------
    # 上下文管理
    # --------------------------------------------------

    def close(self):
        if self.connected:
            self.cli.close()
            self.connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
