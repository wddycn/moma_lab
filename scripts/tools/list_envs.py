# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
功能：
    列出 Isaac Lab 中已经注册的所有环境(task)。
    通过 gymnasium 注册表获取环境名称、入口函数和配置文件。
"""

# 启动 Isaac Sim
import argparse
from isaaclab.app import AppLauncher

# =======================
# 参数配置
# =======================
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
# 可选参数：按照关键词筛选环境
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
# parse the arguments
args_cli = parser.parse_args()

# 启动 Isaac Sim，使用无界面模式
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""
# =======================
# 导入 Moma Lab 环境
# =======================
import textwrap

import gymnasium as gym

# 导入自定义任务，使其注册到 gym 中
import moma_lab.tasks  # noqa: F401
from prettytable import PrettyTable


def main():
    """Print all environments registered in `moma_lab_tasks` extension."""
    # print all the available environments
    # 创建表格
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Moma Lab"
    # 设置表格对齐方式
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    table.hrules = 1

    # 设置每列的最大宽度
    max_width = 50

    # 初始化索引
    index = 0
    # acquire all Moma Lab environments names
    # 遍历 gym 注册表中的所有任务
    for task_spec in gym.registry.values():
        if "RobotLab" in task_spec.id and (args_cli.keyword is None or args_cli.keyword in task_spec.id):
            # wrap long text in each column before adding it to the table
            task_name = textwrap.fill(task_spec.id, max_width)
            entry_point = textwrap.fill(task_spec.entry_point, max_width)
            config = textwrap.fill(task_spec.kwargs["env_cfg_entry_point"], max_width)

            # add details to table
            table.add_row([index + 1, task_name, entry_point, config])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
