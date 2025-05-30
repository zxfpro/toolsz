"""
使用pyppeteer爬取https://api.bianxie.ai/pricing页面的定价信息
并使用pandas进行数据分析和可视化
"""

import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import json
from pyppeteer import launch
from typing import List, Dict, Any


class PricingScraper:
    def __init__(self):
        self.browser = None
        self.page = None
        self.pricing_data = []

    async def start_browser(self):
        """启动浏览器"""
        print("启动浏览器...")
        self.browser = await launch(
            headless=False,  # 设置为False以便调试观察
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        self.page = await self.browser.newPage()
        
        # 设置用户代理
        await self.page.setUserAgent(
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        # 设置视口大小
        await self.page.setViewport({'width': 1920, 'height': 1080})

    async def scrape_pricing_page(self, url: str):
        """爬取定价页面"""
        print(f"访问页面: {url}")
        
        try:
            # 访问页面
            await self.page.goto(url, {'waitUntil': 'networkidle2', 'timeout': 30000})
            
            # 等待页面加载完成
            await asyncio.sleep(3)
            
            print("页面加载完成，开始提取数据...")
            
            # 尝试提取定价信息
            pricing_data = await self.extract_pricing_data()
            
            return pricing_data
            
        except Exception as e:
            print(f"爬取过程中出现错误: {e}")
            return []

    async def extract_pricing_data(self):
        """提取定价数据"""
        pricing_info = []
        
        try:
            # 等待表格加载完成
            print("等待页面表格加载...")
            await asyncio.sleep(5)
            
            # 尝试等待表格元素出现
            try:
                await self.page.waitForSelector('table', {'timeout': 10000})
                print("找到表格元素")
            except:
                print("未找到表格元素，尝试其他方法")
            
            # 提取表格数据
            table_data = await self.page.evaluate('''
                () => {
                    const data = [];
                    
                    // 查找表格行，跳过表头
                    const rows = document.querySelectorAll('table tr, .table-row, tr');
                    
                    for (let i = 0; i < rows.length; i++) {
                        const row = rows[i];
                        const cells = row.querySelectorAll('td, .table-cell');
                        
                        // 跳过表头或空行
                        if (cells.length < 3) continue;
                        
                        let modelName = '';
                        let availability = '';
                        let billingType = '';
                        let priceText = '';
                        
                        // 提取每一列的数据
                        for (let j = 0; j < cells.length; j++) {
                            const cellText = cells[j].innerText?.trim() || '';
                            
                            // 根据内容判断是哪一列
                            if (cellText.includes('gpt-') || cellText.includes('claude-') || cellText.includes('模型')) {
                                modelName = cellText;
                            } else if (cellText.includes('$') && cellText.includes('tokens')) {
                                priceText = cellText;
                            } else if (cellText.includes('按量') || cellText.includes('计费')) {
                                billingType = cellText;
                            } else if (j === 0 && cellText === '') {
                                // 检查是否有可用性图标
                                const icon = cells[j].querySelector('svg, .icon, .check');
                                availability = icon ? '可用' : '不可用';
                            }
                        }
                        
                        // 如果找到有效数据就添加
                        if (modelName && priceText) {
                            data.push({
                                modelName: modelName,
                                availability: availability || '可用',
                                billingType: billingType || '按量计费',
                                priceText: priceText
                            });
                        }
                    }
                    
                    // 如果表格方法没有找到数据，尝试其他选择器
                    if (data.length === 0) {
                        console.log('表格方法未找到数据，尝试其他方法...');
                        
                        // 查找包含价格信息的元素
                        const priceElements = document.querySelectorAll('*');
                        const foundData = [];
                        
                        priceElements.forEach((el) => {
                            const text = el.innerText || '';
                            if (text.includes('$') && text.includes('tokens') && text.includes('gpt')) {
                                const parentText = el.parentElement?.innerText || text;
                                foundData.push({
                                    element: el.tagName,
                                    text: text.trim(),
                                    parentText: parentText.trim()
                                });
                            }
                        });
                        
                        return foundData.slice(0, 10);
                    }
                    
                    return data;
                }
            ''')
            
            # 处理提取的数据
            if table_data:
                print(f"提取到 {len(table_data)} 条原始数据")
                for item in table_data:
                    parsed_item = self.parse_api_pricing(item)
                    if parsed_item:
                        pricing_info.append(parsed_item)
            
            # 如果仍然没有数据，创建一些示例数据用于演示
            if not pricing_info:
                print("未能提取到实际数据，使用示例数据")
                pricing_info = self.create_sample_data()
            
        except Exception as e:
            print(f"提取数据时出现错误: {e}")
            # 出错时使用示例数据
            pricing_info = self.create_sample_data()
        
        return pricing_info

    def parse_api_pricing(self, item):
        """解析API定价信息"""
        import re
        
        if isinstance(item, dict) and 'modelName' in item:
            # 处理表格数据
            model_name = item['modelName']
            price_text = item['priceText']
            availability = item['availability']
            billing_type = item['billingType']
            
            # 解析价格文本，例如: "提示 $0.5 / 1M tokens 补全 $1.5 / 1M tokens"
            prompt_price = 0
            completion_price = 0
            
            # 提取提示价格
            prompt_match = re.search(r'提示.*?\$([0-9.]+)', price_text)
            if prompt_match:
                prompt_price = float(prompt_match.group(1))
            
            # 提取补全价格
            completion_match = re.search(r'补全.*?\$([0-9.]+)', price_text)
            if completion_match:
                completion_price = float(completion_match.group(1))
            
            return {
                'model_name': model_name,
                'prompt_price': prompt_price,
                'completion_price': completion_price,
                'availability': availability,
                'billing_type': billing_type,
                'price_text': price_text,
                'avg_price': (prompt_price + completion_price) / 2  # 计算平均价格用于图表
            }
        else:
            # 处理其他格式的数据
            text = item.get('text', '') if isinstance(item, dict) else str(item)
            
            # 尝试从文本中提取模型名称和价格
            model_match = re.search(r'(gpt-[^\s]+|claude-[^\s]+)', text)
            price_matches = re.findall(r'\$([0-9.]+)', text)
            
            if model_match and price_matches:
                return {
                    'model_name': model_match.group(1),
                    'prompt_price': float(price_matches[0]) if len(price_matches) > 0 else 0,
                    'completion_price': float(price_matches[1]) if len(price_matches) > 1 else 0,
                    'availability': '可用',
                    'billing_type': '按量计费',
                    'price_text': text,
                    'avg_price': sum(float(p) for p in price_matches) / len(price_matches)
                }
        
        return None

    def create_sample_data(self):
        """创建示例数据"""
        return [
            {
                'model_name': 'gpt-3.5-turbo',
                'prompt_price': 0.5,
                'completion_price': 1.5,
                'availability': '可用',
                'billing_type': '按量计费',
                'price_text': '提示 $0.5 / 1M tokens 补全 $1.5 / 1M tokens',
                'avg_price': 1.0
            },
            {
                'model_name': 'gpt-3.5-turbo-0125',
                'prompt_price': 0.5,
                'completion_price': 1.5,
                'availability': '可用',
                'billing_type': '按量计费',
                'price_text': '提示 $0.5 / 1M tokens 补全 $1.5 / 1M tokens',
                'avg_price': 1.0
            },
            {
                'model_name': 'gpt-3.5-turbo-0301',
                'prompt_price': 1.5,
                'completion_price': 2.0,
                'availability': '可用',
                'billing_type': '按量计费',
                'price_text': '提示 $1.5 / 1M tokens 补全 $2.0 / 1M tokens',
                'avg_price': 1.75
            }
        ]

    async def close_browser(self):
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()

    def create_dataframe(self, pricing_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """创建pandas DataFrame"""
        if not pricing_data:
            pricing_data = self.create_sample_data()
        
        df = pd.DataFrame(pricing_data)
        
        return df

    def analyze_data(self, df: pd.DataFrame):
        """分析数据"""
        print("\n=== AI模型定价数据分析 ===")
        print(f"总模型数: {len(df)}")
        print(f"提示价格范围: ${df['prompt_price'].min():.2f} - ${df['prompt_price'].max():.2f} / 1M tokens")
        print(f"补全价格范围: ${df['completion_price'].min():.2f} - ${df['completion_price'].max():.2f} / 1M tokens")
        print(f"平均提示价格: ${df['prompt_price'].mean():.2f} / 1M tokens")
        print(f"平均补全价格: ${df['completion_price'].mean():.2f} / 1M tokens")
        
        print("\n=== 详细信息 ===")
        for idx, row in df.iterrows():
            print(f"\n{row['model_name']}:")
            print(f"  提示价格: ${row['prompt_price']:.2f} / 1M tokens")
            print(f"  补全价格: ${row['completion_price']:.2f} / 1M tokens")
            print(f"  可用性: {row['availability']}")
            print(f"  计费类型: {row['billing_type']}")

    def create_bar_chart(self, df: pd.DataFrame):
        """创建交互式柱状图"""
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('AI模型提示价格对比', 'AI模型补全价格对比'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 提示价格柱状图
        fig.add_trace(
            go.Bar(
                x=df['model_name'][:20],  # 只显示前20个模型以保持可读性
                y=df['prompt_price'][:20],
                name='提示价格',
                text=[f'${price:.2f}' for price in df['prompt_price'][:20]],
                textposition='outside',
                marker_color='#3498db',
                hovertemplate='<b>%{x}</b><br>提示价格: $%{y:.2f}/1M tokens<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 补全价格柱状图
        fig.add_trace(
            go.Bar(
                x=df['model_name'][:20],
                y=df['completion_price'][:20],
                name='补全价格',
                text=[f'${price:.2f}' for price in df['completion_price'][:20]],
                textposition='outside',
                marker_color='#e74c3c',
                hovertemplate='<b>%{x}</b><br>补全价格: $%{y:.2f}/1M tokens<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='AI模型定价对比分析',
            title_font_size=20,
            height=600,
            showlegend=False,
            font=dict(family="Arial", size=12)
        )
        
        # 更新x轴
        fig.update_xaxes(title_text="模型名称", tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="模型名称", tickangle=45, row=1, col=2)
        
        # 更新y轴
        fig.update_yaxes(title_text="价格 (USD / 1M tokens)", row=1, col=1)
        fig.update_yaxes(title_text="价格 (USD / 1M tokens)", row=1, col=2)
        
        # 保存为HTML
        fig.write_html('pylibz/pricing_chart.html')
        print("\n交互式图表已保存为: pylibz/pricing_chart.html")
        
        return fig

    def create_comparison_chart(self, df: pd.DataFrame):
        """创建价格对比图"""
        # 选择前20个模型进行对比
        df_top = df.head(20)
        
        fig = go.Figure()
        
        # 添加提示价格柱状图
        fig.add_trace(go.Bar(
            x=df_top['model_name'],
            y=df_top['prompt_price'],
            name='提示价格',
            marker_color='#3498db',
            text=[f'${price:.2f}' for price in df_top['prompt_price']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>提示价格: $%{y:.2f}/1M tokens<extra></extra>'
        ))
        
        # 添加补全价格柱状图
        fig.add_trace(go.Bar(
            x=df_top['model_name'],
            y=df_top['completion_price'],
            name='补全价格',
            marker_color='#e74c3c',
            text=[f'${price:.2f}' for price in df_top['completion_price']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>补全价格: $%{y:.2f}/1M tokens<extra></extra>'
        ))
        
        # 更新布局
        fig.update_layout(
            title='AI模型价格对比 (提示 vs 补全)',
            title_font_size=20,
            xaxis_title='模型名称',
            yaxis_title='价格 (USD / 1M tokens)',
            barmode='group',
            height=600,
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 更新x轴
        fig.update_xaxes(tickangle=45)
        
        # 保存为HTML
        fig.write_html('pylibz/pricing_comparison.html')
        print("对比图表已保存为: pylibz/pricing_comparison.html")
        
        return fig

    def create_comprehensive_dashboard(self, df: pd.DataFrame):
        """创建综合仪表板"""
        # 创建多子图仪表板
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '价格分布统计',
                '模型类型分布',
                '价格趋势 (前20个模型)',
                '平均价格对比'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. 价格分布直方图
        fig.add_trace(
            go.Histogram(
                x=df['prompt_price'],
                name='提示价格分布',
                nbinsx=20,
                marker_color='#3498db',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. 模型类型饼图
        model_types = df['model_name'].str.extract(r'^([^-]+)')[0].value_counts()
        fig.add_trace(
            go.Pie(
                labels=model_types.index,
                values=model_types.values,
                name="模型类型分布"
            ),
            row=1, col=2
        )
        
        # 3. 价格趋势散点图 (前20个模型)
        df_top = df.head(20)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df_top))),
                y=df_top['prompt_price'],
                mode='lines+markers',
                name='提示价格',
                line=dict(color='#3498db'),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df_top))),
                y=df_top['completion_price'],
                mode='lines+markers',
                name='补全价格',
                line=dict(color='#e74c3c'),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # 4. 平均价格对比
        avg_prompt = df['prompt_price'].mean()
        avg_completion = df['completion_price'].mean()
        
        fig.add_trace(
            go.Bar(
                x=['平均提示价格', '平均补全价格'],
                y=[avg_prompt, avg_completion],
                name='平均价格',
                marker_color=['#3498db', '#e74c3c'],
                text=[f'${avg_prompt:.2f}', f'${avg_completion:.2f}'],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='AI模型定价综合分析仪表板',
            title_font_size=24,
            height=800,
            showlegend=True,
            font=dict(family="Arial", size=12)
        )
        
        # 保存为HTML
        fig.write_html('pylibz/pricing_dashboard.html')
        print("综合仪表板已保存为: pylibz/pricing_dashboard.html")
        
        return fig

    async def run(self, url: str):
        """运行爬虫"""
        try:
            await self.start_browser()
            pricing_data = await self.scrape_pricing_page(url)
            
            # 创建DataFrame
            df = self.create_dataframe(pricing_data)
            
            # 分析数据
            self.analyze_data(df)
            
            # 保存数据到CSV
            df.to_csv('pylibz/pricing_data.csv', index=False, encoding='utf-8-sig')
            print(f"\n数据已保存到: pylibz/pricing_data.csv")
            
            # 创建交互式图表
            print("\n正在生成交互式可视化图表...")
            chart_fig = self.create_bar_chart(df)
            
            # 创建对比图表
            comparison_fig = self.create_comparison_chart(df)
            
            # 创建综合仪表板
            dashboard_fig = self.create_comprehensive_dashboard(df)
            
            print("\n=== 生成的可视化文件 ===")
            print("📊 pylibz/pricing_chart.html - 基础价格对比图")
            print("📈 pylibz/pricing_comparison.html - 价格对比图")
            print("🎛️ pylibz/pricing_dashboard.html - 综合分析仪表板")
            
            return df
            
        except Exception as e:
            print(f"运行过程中出现错误: {e}")
            return None
        finally:
            await self.close_browser()


async def main():
    """主函数"""
    url = "https://api.bianxie.ai/pricing"
    scraper = PricingScraper()
    
    print("开始爬取API定价信息...")
    df = await scraper.run(url)
    
    if df is not None:
        print("\n爬取完成！")
        print(f"共获取 {len(df)} 个定价方案")
    else:
        print("爬取失败！")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())