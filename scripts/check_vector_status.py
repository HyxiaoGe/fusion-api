#!/usr/bin/env python3
"""
检查向量数据库和摘要生成状态
"""

import sys
import os
from datetime import datetime, date, timedelta
# from tabulate import tabulate

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.services.vector_service import VectorService
from app.db.models import HotTopic, DailyTopicDigest
from sqlalchemy import func


def check_vector_status():
    """检查向量数据库状态"""
    db = SessionLocal()
    try:
        vector_service = VectorService(db)
        
        # 获取集合信息
        collection_info = vector_service.client.get_collection(vector_service.collection_name)
        
        print("\n=== Qdrant 向量数据库状态 ===")
        print(f"集合名称: {vector_service.collection_name}")
        print(f"向量数量: {collection_info.points_count}")
        print(f"向量维度: {collection_info.config.params.vectors.size}")
        print(f"距离度量: {collection_info.config.params.vectors.distance}")
        print(f"状态: {collection_info.status}")
        
        # 获取最近的几个向量样本
        if collection_info.points_count > 0:
            result = vector_service.client.scroll(
                collection_name=vector_service.collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            print("\n=== 最近的向量样本 ===")
            for point in result[0]:
                payload = point.payload
                print(f"\n话题: {payload.get('title', 'N/A')[:60]}...")
                print(f"分类: {payload.get('category', 'N/A')}")
                print(f"来源: {payload.get('source', 'N/A')}")
                print(f"创建时间: {payload.get('created_at', 'N/A')}")
        
    except Exception as e:
        print(f"检查向量数据库失败: {e}")
    finally:
        db.close()


def check_hot_topics_status():
    """检查热点话题状态"""
    db = SessionLocal()
    try:
        print("\n=== 热点话题数据状态 ===")
        
        # 总数
        total_count = db.query(func.count(HotTopic.id)).scalar()
        print(f"总话题数: {total_count}")
        
        # 按时间统计
        time_stats = []
        for days in [1, 7, 30]:
            count = db.query(func.count(HotTopic.id)).filter(
                HotTopic.created_at > datetime.now() - timedelta(days=days)
            ).scalar()
            time_stats.append([f"最近{days}天", count])
        
        print("\n时间分布:")
        print(tabulate(time_stats, headers=["时间范围", "话题数"], tablefmt="grid"))
        
        # 按分类统计
        category_stats = db.query(
            HotTopic.category,
            func.count(HotTopic.id).label('count')
        ).group_by(HotTopic.category).order_by(func.count(HotTopic.id).desc()).limit(10).all()
        
        print("\n分类分布 (Top 10):")
        category_table = [[cat or "未分类", count] for cat, count in category_stats]
        print(tabulate(category_table, headers=["分类", "数量"], tablefmt="grid"))
        
        # 按来源统计
        source_stats = db.query(
            HotTopic.source,
            func.count(HotTopic.id).label('count')
        ).group_by(HotTopic.source).order_by(func.count(HotTopic.id).desc()).limit(10).all()
        
        print("\n来源分布 (Top 10):")
        source_table = [[source, count] for source, count in source_stats]
        print(tabulate(source_table, headers=["来源", "数量"], tablefmt="grid"))
        
    except Exception as e:
        print(f"检查热点话题失败: {e}")
    finally:
        db.close()


def check_daily_digests():
    """检查每日摘要状态"""
    db = SessionLocal()
    try:
        print("\n=== 每日摘要状态 ===")
        
        # 获取最近的摘要
        recent_digests = db.query(DailyTopicDigest).order_by(
            DailyTopicDigest.date.desc()
        ).limit(20).all()
        
        if recent_digests:
            # 按日期分组
            digests_by_date = {}
            for digest in recent_digests:
                if digest.date not in digests_by_date:
                    digests_by_date[digest.date] = []
                digests_by_date[digest.date].append(digest)
            
            for digest_date, digests in sorted(digests_by_date.items(), reverse=True):
                print(f"\n日期: {digest_date}")
                digest_table = []
                for digest in digests:
                    digest_table.append([
                        digest.category,
                        digest.cluster_title[:40] + "..." if len(digest.cluster_title) > 40 else digest.cluster_title,
                        digest.topic_count,
                        f"{digest.heat_score:.1f}",
                        digest.view_count
                    ])
                print(tabulate(
                    digest_table,
                    headers=["分类", "标题", "话题数", "热度", "浏览"],
                    tablefmt="grid"
                ))
        else:
            print("还没有生成任何每日摘要")
            
        # 统计信息
        total_digests = db.query(func.count(DailyTopicDigest.id)).scalar()
        print(f"\n总摘要数: {total_digests}")
        
    except Exception as e:
        print(f"检查每日摘要失败: {e}")
    finally:
        db.close()


def main():
    """主函数"""
    print("=" * 60)
    print("向量数据库和摘要系统状态检查")
    print("=" * 60)
    
    check_vector_status()
    check_hot_topics_status()
    check_daily_digests()
    
    print("\n" + "=" * 60)
    print("检查完成！")
    print("\n提示：")
    print("1. 访问 http://localhost:6333/dashboard 查看 Qdrant Web UI")
    print("2. 向量化任务每小时执行一次（30分）")
    print("3. 每日摘要在凌晨2点生成")


if __name__ == "__main__":
    main()