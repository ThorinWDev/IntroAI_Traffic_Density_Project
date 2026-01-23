import tkinter as tk
from tkinter import messagebox
import requests


class MapRouteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("高德地图路径规划器")
        self.api_key = "6f83746aa2de391e7e85c4625aaf654f"  # <--- 在这里填写你的Key

        # UI 布局
        tk.Label(root, text="起点 (经度,纬度):").grid(row=0, column=0, padx=10, pady=5)
        self.origin_ent = tk.Entry(root)
        self.origin_ent.insert(0, "116.481028,39.989643")  # 默认：方恒国际
        self.origin_ent.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(root, text="途径点 (用分号分隔):").grid(row=1, column=0, padx=10, pady=5)
        self.waypoints_ent = tk.Entry(root)
        self.waypoints_ent.insert(0, "116.434446,39.90816")  # 默认：北京站
        self.waypoints_ent.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="终点 (经度,纬度):").grid(row=2, column=0, padx=10, pady=5)
        self.destination_ent = tk.Entry(root)
        self.destination_ent.insert(0, "116.397428,39.90923")  # 默认：天安门
        self.destination_ent.grid(row=2, column=1, padx=10, pady=5)

        self.run_btn = tk.Button(root, text="获取最佳路径", command=self.get_route)
        self.run_btn.grid(row=3, column=0, columnspan=2, pady=10)

        self.result_text = tk.Text(root, height=10, width=40)
        self.result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def get_route(self):
        origin = self.origin_ent.get()
        destination = self.destination_ent.get()
        waypoints = self.waypoints_ent.get()

        # 高德驾车路径规划 API URL
        url = "https://restapi.amap.com/v3/direction/driving"

        params = {
            "key": self.api_key,
            "origin": origin,
            "destination": destination,
            "waypoints": waypoints,
            "extensions": "base",  # 返回基本信息
            "strategy": "10"  # 策略10：最佳优先，不走高速
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if data['status'] == '1':
                # 解析返回的第一条路径
                route = data['route']['paths'][0]
                distance = int(route['distance']) / 1000  # 米转公里
                duration = int(route['duration']) / 60  # 秒转分钟

                res_str = f"规划成功！\n"
                res_str += f"全程距离: {distance:.2f} 公里\n"
                res_str += f"预计耗时: {duration:.1f} 分钟\n"
                res_str += "--------------------\n"
                res_str += "主要路段说明:\n"

                # 获取路段指示
                for step in route['steps'][:5]:  # 只展示前5段
                    res_str += f"- {step['instruction']}\n"

                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, res_str)
            else:
                messagebox.showerror("错误", f"API返回错误: {data['info']}")
        except Exception as e:
            messagebox.showerror("错误", f"请求失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MapRouteApp(root)
    root.mainloop()