#!/usr/bin/env python3
"""
测试 API 接口（包含认证）
"""

import requests
import json
from datetime import date
from getpass import getpass


def login(base_url, username, password):
    """登录获取 token"""
    response = requests.post(
        f"{base_url}/api/auth/login",
        data={
            "username": username,
            "password": password
        }
    )
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        print(f"登录失败: {response.status_code}")
        print(response.json())
        return None


def test_daily_digests(base_url, token):
    """测试每日摘要接口"""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # 获取今天的摘要
    response = requests.get(
        f"{base_url}/api/digests/daily",
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n=== {data['date']} 的每日摘要 ===")
        print(f"总数: {data['total']}")
        
        for digest in data['digests']:
            print(f"\n分类: {digest['category']}")
            print(f"标题: {digest['cluster_title']}")
            print(f"摘要: {digest['cluster_summary'][:100]}..." if digest['cluster_summary'] else "无摘要")
            print(f"话题数: {digest['topic_count']}")
            print(f"热度: {digest['heat_score']}")
            print(f"浏览: {digest['view_count']}")
            if digest['key_points']:
                print("关键点:")
                for point in digest['key_points'][:3]:
                    print(f"  - {point}")
    else:
        print(f"获取摘要失败: {response.status_code}")
        print(response.json())


def test_hot_topics(base_url, token):
    """测试热点话题接口"""
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # 获取热点话题
    response = requests.get(
        f"{base_url}/api/topics/hot?limit=5",
        headers=headers
    )
    
    if response.status_code == 200:
        topics = response.json()
        print(f"\n=== 最新热点话题 (前5条) ===")
        
        for topic in topics:
            print(f"\n标题: {topic['title']}")
            print(f"来源: {topic['source']}")
            print(f"分类: {topic.get('category', '未分类')}")
            print(f"创建时间: {topic['created_at']}")
            print(f"浏览次数: {topic['view_count']}")
    else:
        print(f"获取热点话题失败: {response.status_code}")
        print(response.json())


def main():
    """主函数"""
    base_url = "http://localhost:8000"
    
    print("=== Fusion API 测试工具 ===")
    print(f"API 地址: {base_url}")
    
    # 获取登录凭证
    print("\n请输入登录信息:")
    username = input("用户名: ")
    password = getpass("密码: ")
    
    # 登录
    print("\n正在登录...")
    token = login(base_url, username, password)
    
    if not token:
        print("登录失败，退出")
        return
    
    print("登录成功！")
    
    # 测试接口
    while True:
        print("\n选择要测试的接口:")
        print("1. 每日摘要")
        print("2. 热点话题")
        print("3. 退出")
        
        choice = input("\n请选择 (1-3): ")
        
        if choice == "1":
            test_daily_digests(base_url, token)
        elif choice == "2":
            test_hot_topics(base_url, token)
        elif choice == "3":
            print("退出测试")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()